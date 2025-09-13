from __future__ import annotations

import argparse
from pathlib import Path

from src.config import AppConfig
from src.utils.logging import setup_logging
from src.utils.determinism import set_determinism
from src.data.loader import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="CBRN CLI")
    parser.add_argument("command", choices=["load"], help="Command to execute")
    parser.add_argument("path", nargs="?", help="Path to dataset (for load)")
    parser.add_argument("--config", dest="config", default="configs/default.json", help="Path to config JSON")
    parser.add_argument("--id-salt", dest="id_salt", default=None, help="Override ID salt for hashing (optional)")

    args = parser.parse_args()

    cfg = AppConfig.from_json(args.config)
    logger = setup_logging(cfg.logging.log_dir, cfg.logging.filename, cfg.logging.level)
    set_determinism(
        seed=cfg.determinism.seed,
        cudnn_deterministic=cfg.determinism.cudnn_deterministic,
        cudnn_benchmark=cfg.determinism.cudnn_benchmark,
        cublas_workspace=cfg.determinism.cublas_workspace,
        python_hash_seed=cfg.determinism.python_hash_seed,
        tokenizers_parallelism=cfg.determinism.tokenizers_parallelism,
    )

    if args.command == "load":
        if not args.path:
            parser.error("load requires a dataset path")
        ds = load_dataset(
            args.path,
            csv_mapping=cfg.data.csv_mapping,
            id_salt=(args.id_salt if args.id_salt is not None else cfg.data.id_salt),
        )
        logger.info("Loaded %d records from %s", len(ds), args.path)
        # Print a short preview
        for it in ds[:3]:
            logger.info("%s | %s | choices=%d | answer=%d", it.id, it.question[:50].replace("\n", " "), len(it.choices), it.answer)


if __name__ == "__main__":
    main()
