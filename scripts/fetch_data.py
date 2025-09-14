#!/usr/bin/env python3
"""Fetch and verify datasets from registry."""

import argparse
import hashlib
import importlib
import json
import os
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
REGISTRY = DATA / "registry.yaml"


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_with_progress(url: str, to_path: Path, desc: str = "Downloading") -> None:
    """Download file with progress bar."""
    to_path.parent.mkdir(parents=True, exist_ok=True)

    # Check for HF token if it's a HuggingFace URL
    headers = {}
    if "huggingface.co" in url and os.getenv("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.getenv('HF_TOKEN')}"

    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get("Content-Length", 0))

            print(f"{desc}: {to_path.name}")
            if total_size > 0:
                print(f"Size: {total_size / (1024*1024):.1f} MB")

            with open(to_path, "wb") as f:
                downloaded = 0
                block_size = 8192
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"Progress: {percent:.1f}%", end="\r")

            if total_size > 0:
                print()  # New line after progress
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print("Authentication required. Please set HF_TOKEN environment variable.")
            sys.exit(1)
        raise


def unpack_archive(archive: Path, kind: str, out_dir: Path) -> None:
    """Unpack an archive to output directory."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if kind == "tar.gz":
        with tarfile.open(archive, "r:gz") as t:
            t.extractall(out_dir)
    elif kind == "tar":
        with tarfile.open(archive, "r") as t:
            t.extractall(out_dir)
    elif kind == "zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(out_dir)
    elif kind == "none":
        # For files that don't need unpacking (e.g., single parquet files)
        # Rename to remove .download extension and place in output dir
        import shutil
        # Get the original filename without .download extension
        if archive.name.endswith('.download'):
            original_name = archive.name[:-9]  # Remove .download
            # Guess extension from URL or use parquet as default for WMDP
            if 'parquet' in original_name or not '.' in original_name:
                original_name = original_name.split('.')[0] + '.parquet'
        else:
            original_name = archive.name
        target = out_dir / original_name
        shutil.move(str(archive), str(target))
    else:
        raise ValueError(f"Unknown unpack kind: {kind}")


def load_registry() -> Dict[str, Any]:
    """Load dataset registry."""
    if not REGISTRY.exists():
        print(f"Registry not found: {REGISTRY}")
        sys.exit(1)

    with open(REGISTRY) as f:
        data = yaml.safe_load(f)

    return data.get("datasets", {})


def compute_and_update_checksum(name: str, file_path: Path) -> str:
    """Compute checksum and optionally update registry."""
    checksum = sha256_file(file_path)
    print(f"Computed SHA256: {checksum}")

    # Load registry
    with open(REGISTRY) as f:
        data = yaml.safe_load(f)

    if data["datasets"][name]["sha256"] == "PLACEHOLDER_SHA256_TO_BE_COMPUTED":
        print(f"Updating registry with computed checksum...")
        data["datasets"][name]["sha256"] = checksum

        with open(REGISTRY, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        print(f"Registry updated with checksum for {name}")

    return checksum


def fetch_dataset(name: str, force: bool = False) -> Path:
    """Fetch and verify a dataset."""
    registry = load_registry()

    if name not in registry:
        print(f"Unknown dataset: {name}")
        print(f"Available datasets: {', '.join(registry.keys())}")
        sys.exit(1)

    spec = registry[name]

    # Check if already downloaded
    dataset_dir = RAW / name
    if dataset_dir.exists() and not force:
        print(f"Dataset already exists: {dataset_dir}")
        print("Use --force to re-download")
        return dataset_dir

    # Download
    raw_archive = RAW / f"{name}.download"
    download_with_progress(spec["url"], raw_archive, f"Fetching {name}")

    # Verify checksum
    if spec["sha256"] == "PLACEHOLDER_SHA256_TO_BE_COMPUTED":
        checksum = compute_and_update_checksum(name, raw_archive)
    else:
        checksum = sha256_file(raw_archive)
        expected = spec["sha256"].lower()
        if checksum != expected:
            print(f"Checksum mismatch for {name}:")
            print(f"  Expected: {expected}")
            print(f"  Got:      {checksum}")
            raw_archive.unlink()  # Clean up
            sys.exit(1)
        print("Checksum verified ✓")

    # Unpack
    unpack_kind = spec.get("unpack", "none")
    print(f"Unpacking ({unpack_kind})...")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    unpack_archive(raw_archive, unpack_kind, dataset_dir)

    # Clean up archive if it was unpacked
    if unpack_kind != "none":
        raw_archive.unlink()

    # Verify expected files
    for expected_file in spec.get("expected_files", []):
        path = dataset_dir / expected_file
        if not path.exists():
            print(f"Missing expected file: {path}")
            sys.exit(1)

    print(f"Dataset ready: {dataset_dir}")
    return dataset_dir


def process_dataset(name: str, raw_dir: Path) -> Optional[Path]:
    """Process dataset using adapter if configured."""
    registry = load_registry()
    spec = registry[name]

    adapter_spec = spec.get("process", {}).get("adapter")
    if not adapter_spec:
        return None

    print(f"Processing with adapter: {adapter_spec}")

    try:
        mod_name, func_name = adapter_spec.rsplit(":", 1)
        module = importlib.import_module(mod_name)
        adapter_func = getattr(module, func_name)
    except (ImportError, AttributeError) as e:
        print(f"Failed to load adapter: {e}")
        print("Skipping processing step")
        return None

    # Create output directory
    out_dir = PROC / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run adapter
    try:
        output_path = adapter_func(raw_dir, out_dir)
        print(f"Processed dataset: {output_path}")
        return Path(output_path)
    except Exception as e:
        print(f"Adapter failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Fetch and verify datasets from registry"
    )
    parser.add_argument(
        "name",
        nargs="?",
        help="Dataset name from registry (omit to list available)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if exists"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets"
    )
    parser.add_argument(
        "--compute-checksum",
        metavar="FILE",
        help="Compute SHA256 checksum for a file"
    )

    args = parser.parse_args()

    # Handle checksum computation
    if args.compute_checksum:
        path = Path(args.compute_checksum)
        if not path.exists():
            print(f"File not found: {path}")
            sys.exit(1)
        print(sha256_file(path))
        return

    # Handle list
    if args.list or not args.name:
        registry = load_registry()
        print("Available datasets:")
        for name, spec in registry.items():
            notes = spec.get("notes", "")
            if notes:
                print(f"  {name:20} - {notes}")
            else:
                print(f"  {name}")
        return

    # Fetch dataset
    raw_dir = fetch_dataset(args.name, args.force)

    # Process if adapter configured
    processed_path = process_dataset(args.name, raw_dir)

    # Output result
    result = {
        "dataset": args.name,
        "raw_dir": str(raw_dir),
    }
    if processed_path:
        result["processed"] = str(processed_path)

    print("\nResult:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()