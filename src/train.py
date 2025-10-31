"""Training entry point for YOLO11 segmentation on SIIM-ACR Pneumothorax."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from ultralytics import YOLO

from .dataset_prep import DatasetConfig, Yolo11SegmentationDatasetBuilder
from . import paths


app = typer.Typer(help="Train a YOLO11 segmentation model on the processed dataset.")


def _ensure_dataset_exists(
    dataset_yaml: Path,
    rebuild_dataset: bool,
    train_split: float,
    seed: int,
    min_polygon_area: float,
    simplify_epsilon: float,
) -> Path:
    if dataset_yaml.exists() and not rebuild_dataset:
        return dataset_yaml

    config = DatasetConfig(
        output_dir=dataset_yaml.parent,
        train_split=train_split,
        seed=seed,
        min_polygon_area=min_polygon_area,
        simplify_epsilon=simplify_epsilon,
    )
    builder = Yolo11SegmentationDatasetBuilder(config)
    return builder.prepare()


@app.command()
def train(
    model_weights: str = typer.Option(
        "yolo11m-seg.pt",
        help="Ultralytics checkpoint to start from (e.g. yolo11n-seg.pt)",
    ),
    data_yaml: Path = typer.Option(
        paths.PROCESSED_DIR / "dataset.yaml",
        help="Path to the dataset YAML file",
    ),
    rebuild_dataset: bool = typer.Option(
        False,
        help="Force dataset regeneration before training",
    ),
    train_split: float = typer.Option(0.85, help="Fraction of data to use for training when rebuilding"),
    seed: int = typer.Option(42, help="Random seed for dataset split and training"),
    min_polygon_area: float = typer.Option(10.0, help="Minimum polygon area when rebuilding dataset"),
    simplify_epsilon: float = typer.Option(1.5, help="Douglas-Peucker epsilon for contour simplification"),
    epochs: int = typer.Option(50, help="Number of training epochs"),
    batch_size: int = typer.Option(4, help="Images per batch"),
    imgsz: int = typer.Option(1024, help="Image size for training"),
    device: Optional[str] = typer.Option("0", help="Device spec passed to Ultralytics YOLO"),
    project: Path = typer.Option(
        paths.PROJECT_ROOT / "artifacts" / "runs",
        help="Directory where Ultralytics stores training runs",
    ),
    name: str = typer.Option("yolo11-seg-siim", help="Run name"),
    workers: int = typer.Option(8, help="Number of dataloader workers"),
    patience: int = typer.Option(50, help="Early stopping patience"),
) -> None:
    """Train YOLO11 segmentation model using Ultralytics."""

    dataset_yaml = _ensure_dataset_exists(
        dataset_yaml=data_yaml,
        rebuild_dataset=rebuild_dataset,
        train_split=train_split,
        seed=seed,
        min_polygon_area=min_polygon_area,
        simplify_epsilon=simplify_epsilon,
    )

    project.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_weights)

    train_args = {
        "data": dataset_yaml.as_posix(),
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": imgsz,
        "device": device,
        "project": project.as_posix(),
        "name": name,
        "workers": workers,
        "patience": patience,
        "seed": seed,
    }

    typer.echo("Starting Ultralytics YOLO training...")
    model.train(**train_args)

    run_dir = project / name
    best_model = run_dir / "weights" / "best.pt"
    if best_model.exists():
        typer.echo(f"Training complete. Best weights: {best_model}")
    else:
        typer.echo("Training complete. Check the run directory for outputs.")


if __name__ == "__main__":  # pragma: no cover - CLI
    app()
