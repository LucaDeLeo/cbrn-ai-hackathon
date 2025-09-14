import subprocess
from pathlib import Path


def test_validate_dataset_cli_invalid(tmp_path: Path):
    # Create a minimal invalid JSONL (missing required fields for MCQ)
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": 1}\n')

    # Run the CLI validator
    proc = subprocess.run(
        [
            "python",
            "-m",
            "robustcbrn.cli.validate_dataset",
            "--schema",
            "both",
            str(bad),
        ],
        capture_output=True,
        text=True,
    )

    # Expect schema failure exit code and docs guidance in output
    assert proc.returncode == 4, proc.stdout + proc.stderr
    combined = proc.stdout + proc.stderr
    assert "Schema validation failed" in combined
    assert "docs/USAGE.md#dataset-schema" in combined


def test_validate_dataset_cli_out_of_range_answer(tmp_path: Path):
    # MCQ with 2 choices but answer 'C' (out of range)
    bad = tmp_path / "bad_out_of_range.jsonl"
    bad.write_text('{"id": 1, "question": "Q?", "choices": ["A","B"], "answer": "C"}\n')

    proc = subprocess.run(
        [
            "python",
            "-m",
            "robustcbrn.cli.validate_dataset",
            "--schema",
            "mcq",
            str(bad),
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 4, proc.stdout + proc.stderr
    combined = proc.stdout + proc.stderr
    assert "Schema validation failed" in combined
    assert "answer is out of range" in combined or "docs/USAGE.md#dataset-schema" in combined


def test_validate_dataset_cli_nonstring_choice_items(tmp_path: Path):
    # MCQ with a non-string choice element
    bad = tmp_path / "bad_choice_types.jsonl"
    bad.write_text('{"id": 1, "question": "Q?", "choices": ["A", 2, "C"], "answer": 0}\n')

    proc = subprocess.run(
        [
            "python",
            "-m",
            "robustcbrn.cli.validate_dataset",
            str(bad),
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 4, proc.stdout + proc.stderr
    combined = proc.stdout + proc.stderr
    assert "Schema validation failed" in combined
    assert "choices[1] must be a string" in combined or "docs/USAGE.md#dataset-schema" in combined
