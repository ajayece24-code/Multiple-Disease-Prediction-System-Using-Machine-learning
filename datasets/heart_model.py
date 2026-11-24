"""Thin wrapper forwarding to root train_heart.py."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from train_heart import (  # noqa: E402, F401
    CATEGORY_MAPPINGS,
    FEATURE_COLUMNS,
    build_pipeline,
    load_and_prepare_data,
    train_and_save_model,
)

if __name__ == "__main__":
    train_and_save_model()
