from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Add scripts directory for fetch_data tests
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Always disable model loading - we only use mocks
os.environ["DISABLE_MODEL_LOADING"] = "1"


# ====================
# Data Management Fixtures
# ====================

@pytest.fixture
def mock_wmdp_parquet_data():
    """Create mock WMDP Parquet data for testing."""
    return pd.DataFrame({
        'question': [
            'What is the capital of France?',
            'Which element has atomic number 6?',
            'What is 2 + 2?'
        ],
        'choices': [
            np.array(['London', 'Paris', 'Berlin', 'Madrid']),
            np.array(['Oxygen', 'Nitrogen', 'Carbon', 'Hydrogen']),
            np.array(['3', '4', '5', '6'])
        ],
        'answer': [1, 2, 1]  # 0-based indices
    })


@pytest.fixture
def mock_wmdp_csv_data():
    """Create mock WMDP CSV data for testing."""
    return {
        'headers': ['question', 'choice_a', 'choice_b', 'choice_c', 'choice_d', 'answer', 'topic'],
        'rows': [
            {
                'question': 'What is Python?',
                'choice_a': 'A snake',
                'choice_b': 'A programming language',
                'choice_c': 'A framework',
                'choice_d': 'A database',
                'answer': 'B',
                'topic': 'programming'
            },
            {
                'question': 'What is Git?',
                'choice_a': 'Version control',
                'choice_b': 'Database',
                'choice_c': 'Web framework',
                'choice_d': 'Operating system',
                'answer': 'A',
                'topic': 'tools'
            }
        ]
    }


@pytest.fixture
def mock_registry_data():
    """Create mock registry data for testing."""
    return {
        "datasets": {
            "test_bio": {
                "url": "https://example.com/test_bio.parquet",
                "sha256": "a" * 64,  # Valid 64-char hex
                "license": "CC BY 4.0",
                "unpack": "none",
                "process": {
                    "adapter": "robustcbrn.data.adapters.wmdp:convert_wmdp_parquet_to_jsonl"
                },
                "safe_to_publish": False,
                "notes": "Test biosecurity dataset"
            },
            "test_chem": {
                "url": "https://example.com/test_chem.csv",
                "sha256": "b" * 64,
                "license": "MIT",
                "unpack": "none",
                "process": {
                    "adapter": "robustcbrn.data.adapters.wmdp:convert_wmdp_to_jsonl"
                },
                "safe_to_publish": False,
                "notes": "Test chemistry dataset"
            }
        },
        "config": {
            "cache_dir": "${ROBUSTCBRN_DATA_DIR:-data}",
            "verify_checksums": True,
            "max_download_size_mb": 1000
        }
    }


@pytest.fixture
def mock_jsonl_data():
    """Create mock JSONL evaluation data."""
    return [
        {
            "id": "test_001",
            "question": "What is machine learning?",
            "choices": ["A subset of AI", "A database", "A programming language", "An OS"],
            "answer": "A",
            "metadata": {"domain": "AI", "difficulty": "easy"}
        },
        {
            "id": "test_002",
            "question": "What is Docker?",
            "choices": ["Container platform", "Database", "Language", "Editor"],
            "answer": "A",
            "metadata": {"domain": "DevOps", "difficulty": "medium"}
        },
        {
            "id": "test_003",
            "question": "What is React?",
            "choices": ["Database", "JS library", "Language", "OS"],
            "answer": "B",
            "metadata": {"domain": "Web", "difficulty": "easy"}
        }
    ]


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory structure."""
    data_dir = tmp_path / "data"
    (data_dir / "raw").mkdir(parents=True)
    (data_dir / "processed").mkdir(parents=True)

    # Create a sample registry
    registry = data_dir / "registry.yaml"
    registry_data = {
        "datasets": {
            "test_dataset": {
                "url": "https://example.com/test.parquet",
                "sha256": "PLACEHOLDER_SHA256_TO_BE_COMPUTED",
                "license": "MIT",
                "unpack": "none"
            }
        }
    }
    registry.write_text(yaml.dump(registry_data))

    return data_dir


@pytest.fixture
def mock_parquet_file(tmp_path, mock_wmdp_parquet_data):
    """Create a mock Parquet file."""
    parquet_file = tmp_path / "test_data.parquet"
    mock_wmdp_parquet_data.to_parquet(parquet_file)
    return parquet_file


@pytest.fixture
def mock_csv_file(tmp_path, mock_wmdp_csv_data):
    """Create a mock CSV file."""
    import csv

    csv_file = tmp_path / "test_data.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=mock_wmdp_csv_data['headers'])
        writer.writeheader()
        for row in mock_wmdp_csv_data['rows']:
            writer.writerow(row)

    return csv_file


@pytest.fixture
def mock_jsonl_file(tmp_path, mock_jsonl_data):
    """Create a mock JSONL file."""
    jsonl_file = tmp_path / "test_eval.jsonl"
    with open(jsonl_file, 'w') as f:
        for item in mock_jsonl_data:
            f.write(json.dumps(item) + '\n')

    return jsonl_file


@pytest.fixture
def safe_artifacts_dir(tmp_path):
    """Create a directory with safe artifacts (no raw content)."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    # Safe summary without raw content
    summary = {
        "accuracy": 0.92,
        "total_items": 100,
        "consensus_k": 2,
        "exploitable_count": 30,
        "mcq_accuracy": 0.90,
        "cloze_accuracy": 0.94
    }

    (artifacts / "summary.json").write_text(json.dumps(summary, indent=2))

    # Safe CSV without forbidden columns
    csv_content = "id,accuracy,confidence\n1,0.95,0.88\n2,0.87,0.92\n"
    (artifacts / "results.csv").write_text(csv_content)

    return artifacts


@pytest.fixture
def unsafe_artifacts_dir(tmp_path):
    """Create a directory with unsafe artifacts (contains raw content)."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    # Unsafe summary with raw questions
    summary = {
        "accuracy": 0.92,
        "question": "What is the formula?",  # Forbidden
        "choices": ["A", "B", "C", "D"],     # Forbidden
        "exploitable": True                   # Forbidden per-item label
    }

    (artifacts / "bad_summary.json").write_text(json.dumps(summary, indent=2))

    # Unsafe CSV with exploitable column
    csv_content = "id,accuracy,exploitable\n1,0.95,true\n2,0.87,false\n"
    (artifacts / "bad_results.csv").write_text(csv_content)

    return artifacts


# ====================
# Helper Functions for Tests
# ====================



# ====================
# Model Mocking Fixtures
# ====================

@pytest.fixture(autouse=True)
def mock_models_by_default(monkeypatch):
    """Automatically mock all model operations unless explicitly disabled."""
    if os.environ.get("DISABLE_MODEL_LOADING", "1") == "1":
        from tests.mock_models import patch_all_models
        patch_all_models(monkeypatch)


@pytest.fixture
def mock_transformers(monkeypatch):
    """Fixture to mock transformers models."""
    import sys

    from tests.mock_models import MockModel, MockTokenizer

    mock_transformers = type('MockTransformers', (), {
        'AutoTokenizer': MockTokenizer,
        'AutoModelForCausalLM': MockModel,
    })()

    if "transformers" in sys.modules:
        monkeypatch.setattr("transformers.AutoTokenizer", MockTokenizer)
        monkeypatch.setattr("transformers.AutoModelForCausalLM", MockModel)

    return mock_transformers


@pytest.fixture
def mock_cloze_scoring(monkeypatch):
    """Fixture to mock cloze scoring functions."""
    from tests.mock_models import mock_run_cloze_hf, mock_score_cloze_options

    monkeypatch.setattr(
        "robustcbrn.tasks.cloze_full.score_cloze_options",
        mock_score_cloze_options,
        raising=False
    )
    monkeypatch.setattr(
        "robustcbrn.tasks.cloze_logprob.run_cloze_hf",
        mock_run_cloze_hf,
        raising=False
    )



