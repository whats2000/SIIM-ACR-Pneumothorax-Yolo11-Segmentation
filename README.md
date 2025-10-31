# SIIM-ACR Pneumothorax – YOLO11 Segmentation

Pipeline for converting the SIIM-ACR Pneumothorax DICOM dataset into a YOLO11 segmentation-friendly format, training a segmentation model with Ultralytics, and exporting Stage 2–style submissions.

## Environment

The project is managed with `uv`, but any PEP 621 compatible workflow works.

```powershell
uv sync  # or: pip install -e .
```

Optional extras install the appropriate PyTorch build:

```powershell
uv sync --extra cpu   # CPU only
uv sync --extra cu118 # CUDA 11.8
```

## Dataset preparation

Run the CLI to convert DICOMs + RLE annotations into YOLO-ready PNGs and segmentation labels. The output lives under `data/processed/yolo11` by default and includes a `dataset.yaml` file for Ultralytics.

```powershell
uv run python -m src.dataset_prep build `
  --train-split 0.85 `
  --output-dir data/processed/yolo11
```

Important files created:

- `data/processed/yolo11/images/{train,val,test}/*.png`
- `data/processed/yolo11/labels/{train,val}/*.txt`
- `data/processed/yolo11/dataset.yaml`

The CLI skips work that already exists, so re-running is safe.

## Training

Train a YOLO11 segmentation checkpoint with Ultralytics. The command below starts from the `yolo11m-seg.pt` weights, regenerates the dataset if needed, and writes outputs to `artifacts/runs/yolo11-seg-siim`.

```powershell
uv run python -m src.train train `
  --model-weights yolo11m-seg.pt `
  --epochs 50 `
  --batch-size 4 `
  --imgsz 1024 `
  --device 0
```

Adjust `--device` (e.g. `cpu`) and hyperparameters as required. The best weights end up at `artifacts/runs/<run-name>/weights/best.pt`.

## Prediction & submission

Use the trained weights to score Stage 2 test images and emit a Kaggle-style CSV.

```powershell
uv run python -m src.predict predict `
  --weights artifacts/runs/yolo11-seg-siim/weights/best.pt `
  --output-csv submissions/yolo11_stage2.csv
```

The script loads `data/stage_2_sample_submission.csv`, unions per-image masks, and writes the encoded pixels (or `-1` for negative cases).

## Notebook

`notebooks/siim_pneumothorax_visualization.ipynb` showcases how to rebuild the processed dataset, visualise random training examples, and preview inference results once weights are available.

Launch it with Jupyter Lab or VS Code's notebook UI after installing the project dependencies.
