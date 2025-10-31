"""Utilities for training YOLO11 segmentation models on SIIM-ACR Pneumothorax."""

from importlib.metadata import version, PackageNotFoundError

try:  # pragma: no cover - metadata lookup only at runtime
    __version__ = version("siim-acr-pneumothorax-yolo11-segmentation")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.0.0"

__all__ = ["__version__"]
