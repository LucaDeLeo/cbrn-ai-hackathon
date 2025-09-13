from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Any, Dict, Optional


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_dir: str = "logs"
    filename: str = "app.log"

    def file_path(self) -> Path:
        return Path(self.log_dir) / self.filename


@dataclass
class DeterminismConfig:
    seed: int = 42
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False
    cublas_workspace: str = ":4096:8"
    python_hash_seed: int = 0
    tokenizers_parallelism: bool = False


@dataclass
class DataConfig:
    id_salt: str = ""
    csv_mapping: Optional[Dict[str, str]] = None  # maps internal keys to CSV columns


@dataclass
class AppConfig:
    logging: LoggingConfig = None  # type: ignore[assignment]
    determinism: DeterminismConfig = None  # type: ignore[assignment]
    data: DataConfig = None  # type: ignore[assignment]

    @staticmethod
    def from_json(path: str | Path) -> "AppConfig":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return AppConfig(
            logging=LoggingConfig(**payload.get("logging", {})),
            determinism=DeterminismConfig(**payload.get("determinism", {})),
            data=DataConfig(**payload.get("data", {})),
        )

    def to_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({
                "logging": asdict(self.logging),
                "determinism": asdict(self.determinism),
                "data": asdict(self.data),
            }, f, indent=2, ensure_ascii=False)


# Provide safe defaults via a factory function for top-level config
def default_app_config() -> AppConfig:
    return AppConfig(
        logging=LoggingConfig(),
        determinism=DeterminismConfig(),
        data=DataConfig(),
    )
