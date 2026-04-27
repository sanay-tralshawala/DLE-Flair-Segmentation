# DLE FLAIR Segmentation

Prototype pipeline for 5-class land-cover segmentation on the FLAIR-1 toy dataset.

## Current Status

- EDA notebook creates grouped 5-class train/val/test split CSVs.
- `data/processed/class_map.json` maps original FLAIR labels `1..19` to targets `0..4`.
- `src/data.py` uses FLAIR-style `rasterio` reads for all five source channels.
- `build_dataloaders()` tested on one local batch, needs to be added to placeholder once proven compatible
- `src/train.py` is a lightweight torch training loop with checkpoints and (optional) W&B logging.

## Setup Notes

- `requirements.txt` and `pyproject.toml` available for both pip and uv envs
- W&B is disabled by default. To enable it, set `wandb.enabled: true` in the YAML config.
- For local W&B runs, copy `example.env` to `.env` and set `WANDB_API_KEY`. `.env` is ignored by git.
- For **Google Colab W&B runs**, add a Colab secret named `WANDB_API_KEY`; the training notebook loads it into the environment before training.
- The raw FLAIR toy dataset is not committed; use the download notebook/data instructions before running training.

## Next Steps

- Implement model builders for `in_channels=5`.
- Implement validation metrics in `src/evaluate.py`, starting with `val_loss` and IoU.
- Turn on W&B after the local train/validation path is verified.
