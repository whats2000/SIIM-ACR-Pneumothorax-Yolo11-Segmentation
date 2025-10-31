"""Helper functions for working with DICOM images and pneumothorax masks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2
import numpy as np
import pydicom


@dataclass(slots=True)
class DicomImage:
    """Lightweight container for a DICOM image and its metadata."""

    image_id: str
    array: np.ndarray
    pixel_spacing: Sequence[float] | None = None

    @property
    def height(self) -> int:
        return int(self.array.shape[0])

    @property
    def width(self) -> int:
        return int(self.array.shape[1])


def load_dicom(dicom_path: Path) -> DicomImage:
    """Load a DICOM file into a float32 numpy array.

    The pixel intensities are rescaled using RescaleSlope/RescaleIntercept metadata
    before being returned as a float32 array.
    """

    dataset = pydicom.dcmread(str(dicom_path))
    pixel_array = dataset.pixel_array.astype(np.float32)

    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    if slope != 1.0 or intercept != 0.0:
        pixel_array = pixel_array * slope + intercept

    pixel_spacing = getattr(dataset, "PixelSpacing", None)
    return DicomImage(image_id=dicom_path.stem, array=pixel_array, pixel_spacing=pixel_spacing)


def window_image(image: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    """Perform percentile-based windowing and rescale to [0, 255]."""

    lower = np.percentile(image, low)
    upper = np.percentile(image, high)
    if upper <= lower:
        lower, upper = float(image.min()), float(image.max())

    windowed = np.clip(image, lower, upper)
    windowed -= windowed.min()
    if windowed.max() > 0:
        windowed /= windowed.max()
    windowed = (windowed * 255).astype(np.uint8)
    return windowed


def save_image(array: np.ndarray, output_path: Path) -> None:
    """Save a grayscale numpy array as an image file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if array.ndim == 2:
        array_to_save = array
    else:
        array_to_save = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(str(output_path), array_to_save)


def decode_rle_mask(encoded_pixels: str | float | None, shape: tuple[int, int]) -> np.ndarray:
    """Decode the competition RLE mask into a boolean array.

    The RLE format is: start1 length1 offset2 length2 offset3 length3 ...
    where start1 is 1-indexed, and subsequent values are offsets from the
    end of the previous run.

    Returns an array with shape ``shape`` in Fortran order, matching the original
    Kaggle competition specification.
    """

    if not encoded_pixels or encoded_pixels in {"-1", -1}:  # No pneumothorax
        return np.zeros(shape, dtype=np.uint8)

    try:
        pixel_tokens = [int(tok) for tok in str(encoded_pixels).split()]
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Could not parse RLE mask: {encoded_pixels!r}") from exc

    if len(pixel_tokens) < 2:
        return np.zeros(shape, dtype=np.uint8)

    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    
    # First pair is start (1-indexed) and length
    current_pos = pixel_tokens[0] - 1  # Convert to 0-indexed
    length = pixel_tokens[1]
    mask[current_pos:current_pos + length] = 1
    current_pos += length
    
    # Subsequent pairs are offset and length
    for i in range(2, len(pixel_tokens), 2):
        if i + 1 < len(pixel_tokens):
            offset = pixel_tokens[i]
            length = pixel_tokens[i + 1]
            current_pos += offset
            mask[current_pos:current_pos + length] = 1
            current_pos += length

    return mask.reshape(shape, order="F")


def merge_masks(mask_sequences: Iterable[np.ndarray]) -> np.ndarray:
    """Combine multiple binary masks into a single mask."""

    masks = list(mask_sequences)
    if not masks:
        raise ValueError("merge_masks expects at least one mask")
    combined = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        combined = np.maximum(combined, mask.astype(np.uint8))
    return combined


def mask_to_polygons(mask: np.ndarray, min_area: float = 10.0, epsilon: float = 1.5) -> List[np.ndarray]:
    """Convert a binary mask to a list of polygons suitable for YOLO segmentation."""

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: List[np.ndarray] = []
    for contour in contours:
        if len(contour) < 3:
            continue
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        # Approximate the contour to reduce the number of vertices
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        if len(approx) < 3:
            continue
        polygons.append(approx.reshape(-1, 2))
    return polygons


def polygon_to_yolo(poly: np.ndarray, width: int, height: int) -> List[float]:
    """Normalise polygon coordinates into the YOLO segmentation format list."""

    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError(f"Expected polygon with shape (N, 2), received {poly.shape}")

    normalised = []
    for x, y in poly.astype(np.float32):
        normalised.append(float(np.clip(x / width, 0.0, 1.0)))
        normalised.append(float(np.clip(y / height, 0.0, 1.0)))
    return normalised


def polygons_to_label_lines(polygons: Sequence[np.ndarray], width: int, height: int, class_id: int = 0) -> List[str]:
    """Serialize polygons into YOLO segmentation label strings."""

    lines: List[str] = []
    for polygon in polygons:
        coordinates = polygon_to_yolo(polygon, width=width, height=height)
        if len(coordinates) < 6:
            # Need at least three points
            continue
        line = " ".join([str(class_id)] + [f"{coord:.6f}" for coord in coordinates])
        lines.append(line)
    return lines


def mask_to_rle(mask: np.ndarray) -> str:
    """Encode a binary mask into the competition's run-length encoding format."""

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    flat = mask.T.flatten()  # Transpose to switch to Fortran order
    if flat.sum() == 0:
        return "-1"

    padded = np.concatenate([[0], flat, [0]])
    changes = np.where(padded[1:] != padded[:-1])[0] + 1
    starts = changes[0::2]
    ends = changes[1::2]
    lengths = ends - starts
    return " ".join(f"{start} {length}" for start, length in zip(starts, lengths))


__all__ = [
    "DicomImage",
    "decode_rle_mask",
    "load_dicom",
    "mask_to_rle",
    "merge_masks",
    "mask_to_polygons",
    "polygons_to_label_lines",
    "polygon_to_yolo",
    "save_image",
    "window_image",
]
