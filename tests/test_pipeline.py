from __future__ import annotations

import json
from pathlib import Path
import os
import unittest

from src.config import AppConfig
from src.data.loader import load_dataset
from src.security.anonymizer import make_question_id
from src.utils.determinism import set_determinism
from src.utils.logging import setup_logging


ROOT = Path(__file__).resolve().parents[1]


class TestPipeline(unittest.TestCase):
    def setUp(self) -> None:
        # Ensure logging is configured to a temp file under logs/
        setup_logging("logs", "test.log", level="INFO")
        # Apply determinism controls
        set_determinism(
            seed=123,
            cudnn_deterministic=True,
            cudnn_benchmark=False,
            cublas_workspace=":4096:8",
            python_hash_seed=0,
            tokenizers_parallelism=False,
        )

    def test_jsonl_load_sample(self) -> None:
        sample = ROOT / "data" / "wmdp_bio_sample_100.jsonl"
        self.assertTrue(sample.exists(), "Sample WMDP file missing")
        ds = load_dataset(sample)
        self.assertEqual(len(ds), 100)
        # IDs should be deterministic given the same salt and content
        ids_1 = [x.id for x in ds]
        ds2 = load_dataset(sample)
        self.assertEqual(ids_1, [x.id for x in ds2])

    def test_json_csv_parity(self) -> None:
        # JSON fixture
        jpath = ROOT / "tests" / "fixtures" / "sample_questions.json"
        with open(jpath, "r", encoding="utf-8") as f:
            arr = json.load(f)
        # Write to a temp jsonl
        tmp_jsonl = ROOT / "cache" / "tmp.jsonl"
        tmp_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_jsonl, "w", encoding="utf-8") as f:
            for obj in arr:
                f.write(json.dumps(obj) + "\n")

        djson = load_dataset(tmp_jsonl)

        # CSV fixture
        cpath = ROOT / "tests" / "fixtures" / "sample_questions.csv"
        dcsv = load_dataset(cpath, fmt="csv")

        # Parity check: same size and same fields types
        self.assertEqual(len(djson), len(dcsv))
        self.assertEqual([x.answer for x in djson], [x.answer for x in dcsv])
        self.assertEqual([len(x.choices) for x in djson], [len(x.choices) for x in dcsv])

    def test_anonymizer_stability(self) -> None:
        q = "What is the color of the sky?"
        choices = ["Blue", "Green", "Red", "Yellow"]
        id1 = make_question_id(q, choices, salt="s1")
        id2 = make_question_id(q, choices, salt="s1")
        self.assertEqual(id1, id2)
        id3 = make_question_id(q, choices, salt="s2")
        self.assertNotEqual(id1, id3)

    def test_logging_outputs(self) -> None:
        log_file = ROOT / "logs" / "test.log"
        self.assertTrue(log_file.exists(), "Log file not created")

    def test_env_flags_set(self) -> None:
        # Confirm determinism env is set (as strings)
        self.assertEqual(os.environ.get("CUDNN_DETERMINISTIC"), "1")
        self.assertEqual(os.environ.get("CUDNN_BENCHMARK"), "0")
        self.assertIsNotNone(os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
        self.assertEqual(os.environ.get("PYTHONHASHSEED"), "0")
        self.assertEqual(os.environ.get("TOKENIZERS_PARALLELISM"), "false")


if __name__ == "__main__":
    unittest.main()

