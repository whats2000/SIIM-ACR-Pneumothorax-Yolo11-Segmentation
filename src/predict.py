"""Inference utilities to generate Stage 2 submission files."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import typer
from ultralytics import YOLO
from tqdm import tqdm

from .dataset_prep import DatasetConfig, Yolo11SegmentationDatasetBuilder
from .dicom_utils import mask_to_rle
from . import paths


app = typer.Typer(help="Run inference using a trained YOLO11 segmentation model.")


def _ensure_dataset(
    dataset_dir: Path,
    rebuild_dataset: bool,
    train_split: float,
    seed: int,
    min_polygon_area: float,
    simplify_epsilon: float,
) -> None:
    dataset_yaml = dataset_dir / "dataset.yaml"
    test_dir = dataset_dir / "images" / "test"
    if dataset_yaml.exists() and not rebuild_dataset:
        if test_dir.exists() and any(test_dir.glob("*.png")):
            return

    config = DatasetConfig(
        output_dir=dataset_dir,
        train_split=train_split,
        seed=seed,
        min_polygon_area=min_polygon_area,
        simplify_epsilon=simplify_epsilon,
        build_test_split=True,
    )
    builder = Yolo11SegmentationDatasetBuilder(config)
    builder.prepare()


def _result_to_union_mask(result, mask_threshold: float = 0.5) -> np.ndarray | None:
    if result.masks is None:
        return None
    mask_data = result.masks.data
    if mask_data is None:
        return None
    mask_array = mask_data.clone().cpu().numpy()
    if mask_array.ndim == 2:
        mask_array = mask_array[None, ...]
    binary_masks = mask_array > mask_threshold
    union_mask = np.any(binary_masks, axis=0).astype(np.uint8)
    return union_mask


@app.command()
def predict(
    weights: Path = typer.Option(..., help="Path to trained YOLO weights (best.pt)"),
    sample_submission: Path = typer.Option(paths.STAGE2_SAMPLE_SUBMISSION, help="Sample submission CSV"),
    output_csv: Path = typer.Option(Path("submission.csv"), help="Where to store predictions"),
    dataset_dir: Path = typer.Option(paths.PROCESSED_DIR, help="Processed dataset directory"),
    rebuild_dataset: bool = typer.Option(False, help="Regenerate processed dataset before inference"),
    mask_threshold: float = typer.Option(0.5, help="Threshold applied to predicted masks"),
    conf: float = typer.Option(0.25, help="Confidence threshold for YOLO predictions"),
    iou: float = typer.Option(0.7, help="IoU threshold for NMS"),
    imgsz: int = typer.Option(1024, help="Image size for prediction"),
    batch: int = typer.Option(8, help="Batch size for prediction"),
    device: str = typer.Option("0", help="Device spec for Ultralytics (e.g. '0' or 'cpu')"),
    train_split: float = typer.Option(0.85, help="Train split ratio if dataset rebuilding is required"),
    seed: int = typer.Option(42, help="Random seed if dataset rebuilding is required"),
    min_polygon_area: float = typer.Option(10.0, help="Min polygon area if dataset rebuilding is required"),
    simplify_epsilon: float = typer.Option(1.5, help="Simplification epsilon if dataset rebuilding is required"),
) -> None:
    """Predict pneumothorax masks and create a submission CSV."""

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    _ensure_dataset(
        dataset_dir=dataset_dir,
        rebuild_dataset=rebuild_dataset,
        train_split=train_split,
        seed=seed,
        min_polygon_area=min_polygon_area,
        simplify_epsilon=simplify_epsilon,
    )

    sample_df = pd.read_csv(sample_submission)
    image_ids = sample_df["ImageId"].tolist()
    unique_ids = list(dict.fromkeys(image_ids))  # Preserve order while removing duplicates

    test_image_dir = dataset_dir / "images" / "test"
    sources = []
    for image_id in unique_ids:
        image_path = test_image_dir / f"{image_id}.png"
        if not image_path.exists():
            raise FileNotFoundError(f"Processed test image not found: {image_path}. "
                                    "Run dataset preparation first.")
        sources.append(image_path.as_posix())

    model = YOLO(weights.as_posix())

    typer.echo(f"Running inference on {len(sources)} images...")
    results = model.predict(
        source=sources,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        batch=batch,
        device=device,
        save=False,
        verbose=False,
    )

    predictions: Dict[str, str] = {}
    for result in tqdm(results, desc="Encoding masks"):
        image_path = Path(result.path)
        image_id = image_path.stem
        mask = _result_to_union_mask(result, mask_threshold=mask_threshold)
        if mask is None or mask.sum() == 0:
            predictions[image_id] = "-1"
        else:
            predictions[image_id] = mask_to_rle(mask)

    submission_rows = []
    for image_id in image_ids:
        submission_rows.append({
            "ImageId": image_id,
            "EncodedPixels": predictions.get(image_id, "-1"),
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(output_csv, index=False)
    typer.echo(f"Submission saved to {output_csv}")


if __name__ == "__main__":  # pragma: no cover - CLI
    app()
