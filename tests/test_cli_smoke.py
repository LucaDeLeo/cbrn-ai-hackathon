from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestCliSmoke(unittest.TestCase):
    def test_cli_load_smoke(self) -> None:
        # Prepare a temp config to avoid touching default log file
        tmp_cfg = ROOT / "cache" / "cli-smoke-config.json"
        tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg_payload = {
            "logging": {"level": "INFO", "log_dir": "logs", "filename": "cli-smoke.log"},
            "determinism": {
                "seed": 42,
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
                "cublas_workspace": ":4096:8",
                "python_hash_seed": 0,
                "tokenizers_parallelism": False,
            },
            "data": {"id_salt": "", "csv_mapping": None},
        }
        with open(tmp_cfg, "w", encoding="utf-8") as f:
            json.dump(cfg_payload, f)

        dataset = ROOT / "data" / "wmdp_bio_sample_100.jsonl"
        self.assertTrue(dataset.exists(), "Dataset for smoke test missing")

        proc = subprocess.run(
            [sys.executable, str(ROOT / "cli.py"), "load", str(dataset), "--config", str(tmp_cfg), "--id-salt", "test-salt"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        # Should succeed and print a 'Loaded N records' log line to console handler
        combined = (proc.stdout or "") + (proc.stderr or "")
        self.assertEqual(proc.returncode, 0, msg=f"CLI failed: {combined}")
        self.assertIn("Loaded 100 records", combined)

        # And log file should be created
        log_file = ROOT / "logs" / "cli-smoke.log"
        self.assertTrue(log_file.exists(), "CLI log file not created")
        with open(log_file, "r", encoding="utf-8") as f:
            log_text = f.read()
        self.assertIn("Loaded 100 records", log_text)


if __name__ == "__main__":
    unittest.main()
