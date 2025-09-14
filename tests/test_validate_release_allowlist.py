import os
import subprocess
from pathlib import Path


def test_validate_release_allowlist_csv_headers(tmp_path: Path):
    # Create isolated working dir with only a safe artifacts CSV
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    csv = artifacts / "test.csv"
    # Include a 'question' header which is normally forbidden but will be allowed via env
    csv.write_text("id,question,score\n1,foo,0.5\n")

    env = os.environ.copy()
    env["VALIDATE_SAFE_COLUMNS"] = "question"
    # Provide a dataset path for the QA sub-step; point to the repo sample
    env["DATASET"] = str(Path.cwd() / "data" / "sample_sanitized.jsonl")

    # Run the script from within tmp_path so it scans our artifacts dir
    proc = subprocess.run(
        ["bash", str(Path.cwd() / "scripts" / "validate_release.sh")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )

    # Exit code 0 indicates the allowlist was honored and QA passed on sample dataset
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_validate_release_exploitable_never_allowlisted(tmp_path: Path):
    # Create artifacts with a CSV that includes a forbidden 'exploitable' header
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    csv = artifacts / "leaky.csv"
    csv.write_text("id,exploitable,score\n1,0,0.1\n")

    env = os.environ.copy()
    # Attempt to allowlist 'exploitable' (should NOT be allowed)
    env["VALIDATE_SAFE_COLUMNS"] = "exploitable"
    env["DATASET"] = str(Path.cwd() / "data" / "sample_sanitized.jsonl")

    proc = subprocess.run(
        ["bash", str(Path.cwd() / "scripts" / "validate_release.sh")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )

    # Expect policy content failure (exit code 2)
    assert proc.returncode == 2, proc.stdout + proc.stderr
