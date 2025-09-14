"""
Data fetching utilities and CLI used by tests and pipeline scripts.

This module intentionally provides a minimal, dependency-light implementation
based on urllib and stdlib so tests can reliably mock network calls.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Default project paths (tests can monkeypatch these)
ROOT: Path = Path(__file__).resolve().parents[1]
DATA: Path = ROOT / "data"
RAW: Path = DATA / "raw"
PROC: Path = DATA / "processed"
REGISTRY: Path = DATA / "registry.yaml"


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    p = Path(path)
    with open(p, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_with_progress(url: str, target: str | Path) -> None:
    """Download a URL to a local path using urllib, with optional HF auth.

    - Honors HF_TOKEN for huggingface.co URLs via Authorization: Bearer ...
    - Raises urllib.error.URLError on network errors
    - Exits with SystemExit(1) for HTTP 401/403 authentication failures
    """
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    headers = {}
    token = os.getenv("HF_TOKEN")
    if token and ("huggingface.co" in url):
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req) as resp:  # noqa: S310 (trusted by tests)
            # Content-Length may be absent
            total = resp.headers.get("Content-Length")
            total_size = int(total) if total else None

            with open(target_path, "wb") as out:
                # Stream read in chunks
                while True:
                    data = resp.read(8192)
                    if not data:
                        break
                    out.write(data)
    except urllib.error.HTTPError as e:  # Authentication or access errors
        if e.code in (401, 403):
            print(f"Authentication error downloading {url}: {e.reason}")
            raise SystemExit(1)
        # Re-raise other HTTP errors for tests to handle if needed
        raise


def unpack_archive(archive: str | Path, kind: str, out_dir: str | Path) -> None:
    """Unpack supported archives into out_dir.

    kind:
      - 'tar.gz' -> extract
      - 'zip' -> extract
      - 'none' -> no unpacking; if file ends with .download, rename to .parquet
    """
    archive_path = Path(archive)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if kind == "tar.gz":
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(out)
        return

    if kind == "zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(out)
        return

    if kind == "none":
        # For convenience with parquet downloads saved as .download
        if archive_path.suffix == ".download":
            dest = out / (archive_path.stem + ".parquet")
            archive_path.replace(dest)
        else:
            # Simply move into the out directory preserving name
            archive_path.replace(out / archive_path.name)
        return

    raise ValueError("Unknown unpack kind: {kind}")


def load_registry() -> Dict[str, Dict[str, Any]]:
    path = Path(REGISTRY)
    if not path.exists():
        print(f"Registry not found: {path}")
        raise SystemExit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    datasets = data.get("datasets")
    if not isinstance(datasets, dict):
        print("Invalid registry format: missing 'datasets'")
        raise SystemExit(1)
    return datasets


def compute_and_update_checksum(name: str, file_path: str | Path) -> str:
    """Compute sha256 for a file and update registry placeholder if present."""
    checksum = sha256_file(file_path)
    # Update registry in place if placeholder present
    path = Path(REGISTRY)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {"datasets": {}}
        ds = data.get("datasets", {}).get(name)
        if isinstance(ds, dict) and ds.get("sha256") == "PLACEHOLDER_SHA256_TO_BE_COMPUTED":
            ds["sha256"] = checksum
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f)
    return checksum


def fetch_dataset(name: str, force: bool = False) -> Path:
    """Fetch dataset by name based on registry spec.

    Returns the raw dataset directory path.
    """
    dataset_dir = Path(RAW) / name
    if dataset_dir.exists() and not force:
        return dataset_dir

    specs = load_registry()
    if name not in specs:
        print(f"Unknown dataset: {name}")
        raise SystemExit(1)
    spec = specs[name] or {}

    url: str = spec.get("url", f"https://example.com/{name}.tar.gz")
    sha: Optional[str] = spec.get("sha256")
    unpack_kind: str = spec.get("unpack", "tar.gz")
    expected_files = spec.get("expected_files") or []

    # Target archive path in RAW root (file name based on dataset name)
    archive_path = Path(RAW) / name
    Path(RAW).mkdir(parents=True, exist_ok=True)

    # Perform download; allow URLError to bubble for tests
    print(f"Fetching dataset: {name}")
    download_with_progress(url, archive_path)

    # Verify checksum if provided
    if sha is not None:
        actual = sha256_file(archive_path)
        if actual != sha:
            print(f"Checksum mismatch for {name}: expected {sha}, got {actual}")
            raise SystemExit(1)

    # Unpack/move into dataset_dir
    unpack_archive(archive_path, unpack_kind, dataset_dir)

    # Validate expected files
    missing = [p for p in expected_files if not (dataset_dir / p).exists()]
    if missing:
        print(f"Missing expected files for {name}: {missing}")
        raise SystemExit(1)

    return dataset_dir


def process_dataset(name: str, raw_dir: Path) -> Optional[Path]:
    """Run processing adapter if configured; returns output path or None."""
    specs = load_registry()
    conf = specs.get(name, {})
    proc = conf.get("process") if isinstance(conf, dict) else None
    if not proc or "adapter" not in proc:
        return None

    adapter = proc["adapter"]
    try:
        mod_name, func_name = adapter.split(":", 1)
    except ValueError:
        print(f"Invalid adapter spec: {adapter}")
        return None

    try:
        module = importlib.import_module(mod_name)
        func = getattr(module, func_name)
    except Exception as e:
        print(f"Failed to import adapter {adapter}: {e}")
        return None

    out_dir = Path(PROC) / name
    try:
        result = func(Path(raw_dir), out_dir)
        return Path(result) if result is not None else None
    except Exception as e:
        print(f"Adapter {adapter} failed: {e}")
        return None


def list_datasets_text() -> str:
    """Render a simple list of datasets from the registry for CLI."""
    specs = load_registry()
    lines = []
    for k, v in specs.items():
        note = v.get("notes") if isinstance(v, dict) else None
        if note:
            lines.append(f"{k}: {note}")
        else:
            lines.append(k)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Data fetching utility")
    parser.add_argument("dataset", nargs="?", help="Dataset name to fetch/process")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--compute-checksum", dest="compute_checksum", help="Compute SHA256 of file")

    args = parser.parse_args(argv)

    if args.list:
        print(list_datasets_text())
        return 0

    if args.compute_checksum:
        print(sha256_file(args.compute_checksum))
        return 0

    if args.dataset:
        # Fetch and process
        ds_dir = fetch_dataset(args.dataset, force=args.force)
        out = process_dataset(args.dataset, ds_dir)
        if out:
            print(f"Result: {out}")
        else:
            print(f"Fetched {args.dataset} at {ds_dir}")
        return 0

    # If nothing specified, print help
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
