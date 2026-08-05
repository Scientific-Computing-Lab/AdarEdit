#!/usr/bin/env python3
"""Launch the repository's canonical single-model training pipeline."""

from pathlib import Path
import runpy


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


if __name__ == "__main__":
    runpy.run_path(
        str(REPOSITORY_ROOT / "code" / "train_strict_long.py"),
        run_name="__main__",
    )
