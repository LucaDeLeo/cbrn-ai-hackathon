#!/usr/bin/env python3
from __future__ import annotations

import sys


def check_versions() -> None:
    print(f"Python: {sys.version.split()[0]}")
    for pkg in ["torch", "transformers", "numpy", "tqdm"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            print(f"{pkg}: {ver}")
        except Exception as e:
            print(f"{pkg}: NOT INSTALLED ({e.__class__.__name__})")


if __name__ == "__main__":
    check_versions()
