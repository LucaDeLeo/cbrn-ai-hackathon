from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import unittest

from src.config import AppConfig, LoggingConfig, DeterminismConfig, DataConfig, default_app_config


ROOT = Path(__file__).resolve().parents[1]


class TestConfig(unittest.TestCase):
    def test_round_trip(self) -> None:
        cfg = AppConfig(
            logging=LoggingConfig(level="DEBUG", log_dir="logs", filename="roundtrip.log"),
            determinism=DeterminismConfig(
                seed=7,
                cudnn_deterministic=True,
                cudnn_benchmark=False,
                cublas_workspace=":16:8",
                python_hash_seed=1,
                tokenizers_parallelism=True,
            ),
            data=DataConfig(
                id_salt="abc",
                csv_mapping={
                    "question": "q",
                    "choices": ["a", "b", "c", "d"],
                    "answer": "ans",
                    "topic": "topic",
                    "difficulty": "difficulty",
                },
            ),
        )

        out = ROOT / "cache" / "test_config_roundtrip.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        cfg.to_json(out)

        loaded = AppConfig.from_json(out)
        self.assertEqual(asdict(cfg), asdict(loaded))

    def test_default_factory(self) -> None:
        cfg = default_app_config()
        self.assertEqual(cfg.logging.level, "INFO")
        self.assertEqual(cfg.logging.log_dir, "logs")
        self.assertEqual(cfg.logging.filename, "app.log")
        self.assertEqual(cfg.determinism.seed, 42)
        self.assertTrue(cfg.determinism.cudnn_deterministic)
        self.assertFalse(cfg.determinism.cudnn_benchmark)
        self.assertEqual(cfg.determinism.cublas_workspace, ":4096:8")
        self.assertEqual(cfg.determinism.python_hash_seed, 0)
        self.assertFalse(cfg.determinism.tokenizers_parallelism)
        self.assertEqual(cfg.data.id_salt, "")
        self.assertIsNone(cfg.data.csv_mapping)


if __name__ == "__main__":
    unittest.main()

