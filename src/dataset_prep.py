"""Convert SIIM-ACR Pneumothorax DICOMs into a YOLO11 segmentation dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import typer
from tqdm import tqdm

from . import paths
from .dicom_utils import (
    decode_rle_mask,
    load_dicom,
    mask_to_polygons,
    merge_masks,
    polygons_to_label_lines,
    save_image,
    window_image,
)


@dataclass(slots=True)
class DatasetConfig:
    """Configuration for dataset preparation."""

    train_dicom_dir: Path = paths.RAW_TRAIN_DIR
    test_dicom_dir: Path = paths.RAW_TEST_DIR
    annotations_csv: Path = paths.TRAIN_RLE_PATH
    output_dir: Path = paths.PROCESSED_DIR
    train_split: float = 0.85
    seed: int = 42
    class_id: int = 0
    class_name: str = "pneumothorax"
    min_polygon_area: float = 10.0
    simplify_epsilon: float = 1.5
    build_test_split: bool = True


class Yolo11SegmentationDatasetBuilder:
    """Create a YOLO-ready dataset from the SIIM-ACR pneumothorax data."""

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self.images_dir = self.config.output_dir / "images"
        self.labels_dir = self.config.output_dir / "labels"
        self.dicom_paths: Dict[str, Path] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def prepare(self) -> Path:
        """Run the full preparation pipeline and return the dataset YAML path."""

        self._validate_inputs()
        self._create_directories()
        self._build_dicom_index()

        annotations = self._load_annotations()
        train_ids, val_ids = self._train_val_split(annotations)

        train_stats = self._process_subset(train_ids, annotations, subset="train")
        val_stats = self._process_subset(val_ids, annotations, subset="val")

        test_images = 0
        if self.config.build_test_split and self.config.test_dicom_dir.exists():
            test_images = self._process_test_split()

        dataset_yaml_path = self._write_dataset_yaml()
        self._write_summary_json(
            train_stats=train_stats,
            val_stats=val_stats,
            test_images=test_images,
        )
        return dataset_yaml_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_inputs(self) -> None:
        if not self.config.train_dicom_dir.exists():
            raise FileNotFoundError(f"Train DICOM directory not found: {self.config.train_dicom_dir}")
        if not self.config.annotations_csv.exists():
            raise FileNotFoundError(f"Annotation CSV not found: {self.config.annotations_csv}")
        if self.config.build_test_split and not self.config.test_dicom_dir.exists():
            raise FileNotFoundError(f"Test DICOM directory not found: {self.config.test_dicom_dir}")

    def _create_directories(self) -> None:
        for subset in ("train", "val", "test"):
            (self.images_dir / subset).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / subset).mkdir(parents=True, exist_ok=True)

    def _build_dicom_index(self) -> None:
        """Build a mapping from DICOM image_id to file path."""
        from .dicom_utils import load_dicom
        for dicom_path in tqdm(self.config.train_dicom_dir.glob("**/*.dcm"), desc="Indexing DICOM files"):
            try:
                dicom_image = load_dicom(dicom_path)
                self.dicom_paths[dicom_image.image_id] = dicom_path
            except Exception as e:
                typer.echo(f"Failed to load DICOM {dicom_path}: {e}")

    def _load_annotations(self) -> Dict[str, List[str]]:
        df = pd.read_csv(self.config.annotations_csv)
        df.columns = df.columns.str.strip()

        grouped: Dict[str, List[str]] = {}
        for image_id, group in df.groupby("ImageId"):
            encoded_masks: List[str] = []
            for encoded in group["EncodedPixels"].tolist():
                if pd.isna(encoded):
                    encoded_masks.append("")
                else:
                    encoded_masks.append(str(encoded))
            grouped[image_id] = encoded_masks

        # Filter to only include image_ids with existing DICOM files
        filtered_grouped: Dict[str, List[str]] = {}
        for image_id, masks in grouped.items():
            if image_id in self.dicom_paths:
                filtered_grouped[image_id] = masks
        return filtered_grouped

    def _train_val_split(self, annotations: Dict[str, List[str]]) -> tuple[List[str], List[str]]:
        rng = np.random.default_rng(self.config.seed)

        positives: List[str] = []
        negatives: List[str] = []
        for image_id, masks in annotations.items():
            has_mask = any(mask not in {"", "-1"} for mask in masks)
            if has_mask:
                positives.append(image_id)
            else:
                negatives.append(image_id)

        positives = list(rng.permutation(positives))
        negatives = list(rng.permutation(negatives))

        def split(ids: Sequence[str]) -> tuple[List[str], List[str]]:
            if not ids:
                return [], []
            split_index = int(len(ids) * self.config.train_split)
            split_index = max(1, min(split_index, len(ids) - 1)) if len(ids) > 1 else len(ids)
            return list(ids[:split_index]), list(ids[split_index:])

        train_pos, val_pos = split(positives)
        train_neg, val_neg = split(negatives)

        train_ids = list(train_pos + train_neg)
        val_ids = list(val_pos + val_neg)

        rng.shuffle(train_ids)
        rng.shuffle(val_ids)

        return train_ids, val_ids

    def _process_subset(
        self,
        image_ids: Iterable[str],
        annotations: Dict[str, List[str]],
        subset: str,
    ) -> dict:
        subset = subset.lower()
        images_out = self.images_dir / subset
        labels_out = self.labels_dir / subset

        positives = 0
        total = 0

        for image_id in tqdm(list(image_ids), desc=f"Processing {subset}"):
            image_output_path = images_out / f"{image_id}.png"
            label_output_path = labels_out / f"{image_id}.txt"

            if image_output_path.exists() and label_output_path.exists():
                label_text = label_output_path.read_text(encoding="utf-8").strip()
                if label_text:
                    positives += 1
                total += 1
                continue

            dicom_path = self.dicom_paths[image_id]

            dicom_image = load_dicom(dicom_path)
            pixel_array = window_image(dicom_image.array)
            save_image(pixel_array, image_output_path)

            mask_tokens = annotations.get(image_id, [])
            masks = [
                decode_rle_mask(mask, shape=pixel_array.shape)
                for mask in mask_tokens
                if mask not in {"", "-1"}
            ]

            if masks:
                combined_mask = merge_masks(masks)
                polygons = mask_to_polygons(
                    combined_mask,
                    min_area=self.config.min_polygon_area,
                    epsilon=self.config.simplify_epsilon,
                )
                label_lines = polygons_to_label_lines(
                    polygons,
                    width=dicom_image.width,
                    height=dicom_image.height,
                    class_id=self.config.class_id,
                )
            else:
                label_lines = []

            label_output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(label_output_path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(label_lines))

            if label_lines:
                positives += 1
            total += 1

        return {
            "subset": subset,
            "total_images": total,
            "positive_images": positives,
            "negative_images": total - positives,
        }

    def _process_test_split(self) -> int:
        test_dir = self.images_dir / "test"
        dicom_paths = sorted(self.config.test_dicom_dir.glob("**/*.dcm"))

        for dicom_path in tqdm(dicom_paths, desc="Processing test"):
            dicom_image = load_dicom(dicom_path)
            pixel_array = window_image(dicom_image.array)
            image_output_path = test_dir / f"{dicom_image.image_id}.png"
            if image_output_path.exists():
                continue
            save_image(pixel_array, image_output_path)
        return len(dicom_paths)

    def _write_dataset_yaml(self) -> Path:
        yaml_path = self.config.output_dir / "dataset.yaml"
        yaml_content = (
            "# Auto-generated dataset definition for Ultralytics YOLO11\n"
            f"path: {self.config.output_dir.as_posix()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "names:\n"
            f"  {self.config.class_id}: {self.config.class_name}\n"
        )
        yaml_path.write_text(yaml_content, encoding="utf-8")
        return yaml_path

    def _write_summary_json(self, train_stats: dict, val_stats: dict, test_images: int) -> None:
        summary_path = self.config.output_dir / "dataset_summary.json"
        summary = {
            "train": train_stats,
            "val": val_stats,
            "test": {"total_images": int(test_images)} if test_images else {},
            "config": {
                "train_split": self.config.train_split,
                "seed": self.config.seed,
                "min_polygon_area": self.config.min_polygon_area,
                "simplify_epsilon": self.config.simplify_epsilon,
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main(
    command: str = typer.Argument(
        "build",
        metavar="[build]",
        help="Optional compatibility subcommand; always leave as 'build'.",
    ),
    train_dicom_dir: Path = typer.Option(paths.RAW_TRAIN_DIR, help="Directory with train DICOMs"),
    test_dicom_dir: Path = typer.Option(paths.RAW_TEST_DIR, help="Directory with test DICOMs"),
    annotations_csv: Path = typer.Option(paths.TRAIN_RLE_PATH, help="CSV with RLE-encoded masks"),
    output_dir: Path = typer.Option(paths.PROCESSED_DIR, help="Output directory for YOLO assets"),
    train_split: float = typer.Option(0.85, help="Proportion of images used for training"),
    seed: int = typer.Option(42, help="Random seed for the train/val split"),
    min_polygon_area: float = typer.Option(10.0, help="Minimum contour area to keep"),
    simplify_epsilon: float = typer.Option(1.5, help="Douglas-Peucker epsilon for contour simplification"),
    build_test_split: bool = typer.Option(True, help="Whether to convert the test split images"),
) -> None:
    """CLI entry point that wraps :class:`Yolo11SegmentationDatasetBuilder`."""

    if command.lower() != "build":
        typer.echo(f"Unknown command: {command}", err=True)
        raise typer.Exit(code=2)

    config = DatasetConfig(
        train_dicom_dir=train_dicom_dir,
        test_dicom_dir=test_dicom_dir,
        annotations_csv=annotations_csv,
        output_dir=output_dir,
        train_split=train_split,
        seed=seed,
        min_polygon_area=min_polygon_area,
        simplify_epsilon=simplify_epsilon,
        build_test_split=build_test_split,
    )

    builder = Yolo11SegmentationDatasetBuilder(config)
    dataset_yaml = builder.prepare()
    typer.echo(f"Dataset ready -> {dataset_yaml}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    typer.run(main)
