"""Centralised path helpers for the project."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_TRAIN_DIR = DATA_DIR / "dicom-images-train"
RAW_TEST_DIR = DATA_DIR / "dicom-images-test"
TRAIN_RLE_PATH = DATA_DIR / "train-rle.csv"
STAGE2_SAMPLE_SUBMISSION = DATA_DIR / "stage_2_sample_submission.csv"

# Default location for processed YOLO-friendly assets
PROCESSED_DIR = DATA_DIR / "processed" / "yolo11"


def ensure_directories() -> None:
    """Create directories that we expect to exist when running the pipeline."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

