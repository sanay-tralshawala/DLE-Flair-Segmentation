# DLE FLAIR Segmentation

A deep learning project for semantic segmentation of land cover using the [FLAIR dataset](https://github.com/torchgeo/torchgeo), comparing multiple state-of-the-art backbones and exploring techniques like domain adaptation and conformal prediction.

## Project Overview

**Problem/Question:** 
This repository investigates how different deep learning architectures perform on semantic segmentation of aerial imagery for land cover classification. Specifically, it compares:
- **Traditional CNNs:** ResNet34-UNet
- **Modern Efficient CNNs:** ConvNeXt-Tiny
- **Vision Foundation Models:** DINOv3 + ConvNeXt-Tiny

The goal is to identify which architectures provide the best balance of accuracy, efficiency, and training stability for multi-class land cover segmentation tasks, including techniques for uncertainty quantification through conformal prediction.

## Key Features

- **Multiple Model Architectures:** Pre-trained backbones from TIMM and Hugging Face
- **FLAIR Dataset Support:** Works with both full and toy datasets for experimentation
- **Flexible Configuration System:** YAML-based configs for reproducible experiments
- **Advanced Training Techniques:**
  - Class-weighted loss for handling imbalanced data
  - Differential learning rates (lower LR for backbone, higher for head)
  - Backbone freezing/unfreezing strategies
  - Data augmentation (flips, rotations)
- **Uncertainty Quantification:** Conformal prediction for confidence intervals on predictions
- **Comprehensive Evaluation:** IoU, accuracy, precision, recall metrics per class
- **Weights & Biases Integration:** Experiment tracking and visualization

## Project Structure

```
├── notebooks/              # End-to-end workflows
│   ├── 00_download_toy_dataset.ipynb      # Download FLAIR toy dataset
│   ├── 01_eda.ipynb                       # Exploratory data analysis
│   ├── 02-04_train_*.ipynb                # Training notebooks for each model
│   ├── 05_conformal_prediction.ipynb      # Uncertainty quantification
│   ├── 06_test_metrics.ipynb              # Detailed evaluation
│   ├── 07_channel_comparison.ipynb        # Ablation study (3 vs 5 channels)
│   └── 08_reproduce_results.ipynb         # Reproduce key results
├── src/                    # Core library
│   ├── data.py            # Dataset, dataloaders, augmentation
│   ├── models.py          # Model architectures
│   ├── train.py           # Training loop
│   ├── evaluate.py        # Evaluation metrics
│   ├── utils.py           # Utilities
│   └── visualize.py       # Visualization helpers
├── configs/               # YAML configuration files
│   ├── default.yaml       # Shared defaults
│   ├── resnet34_unet_*.yaml
│   ├── convnext_tiny_*.yaml
│   ├── dinov3_convnext_tiny_*.yaml
│   └── prototype_configs/ # Archive of prototype experiments
├── data/
│   └── README.md          # Data setup instructions
├── requirements.txt       # Dependencies
└── pyproject.toml        # Project metadata
```

## Environment Setup (Google Colab)

**Prerequisites:**
- Google account with access to Google Colab (free tier works)
- GPU runtime enabled (T4 or better recommended for training)

### Setup Instructions

1. **Upload repository to Colab:**
   - Mount your Google Drive in Colab
   - Clone the repo or upload as a ZIP file
   
2. **In the first Colab cell, install dependencies:**
   ```python
   !pip install -r requirements.txt
   ```

3. **Add Colab secrets for API keys:**
   - Click the 🔑 **Secrets** icon in the left sidebar
   - Add `WANDB_API_KEY` — For experiment tracking (get from [wandb.ai](https://wandb.ai))
   - Add `HF_TOKEN` — For Hugging Face gated models like DINOv3 (get from [huggingface.co](https://huggingface.co/settings/tokens))
   - The notebooks will automatically load these via `userdata.get_secret()`

4. **Enable GPU (required for training):**
   - Go to **Runtime** → **Change runtime type** → Select **GPU** (T4 or L4 recommended)

## Data Setup

**Repository Size:** ~50 MB (code + configs only)

### Option A: Toy Dataset (Fastest, ~1.4 GB)

Perfect for prototyping and testing:

```python
# In Colab notebook cell:
!python -m jupyter nbconvert --to notebook --execute notebooks/00_download_toy_dataset.ipynb
```

This downloads and extracts the FLAIR toy dataset into `data/raw/` (~1.4 GB). Then run `01_eda.ipynb` to generate the class map and train/val/test splits.

**Storage requirements after setup:**
- `data/raw/` (toy dataset): ~1.4 GB
- `data/processed/` (processed dataset): ~0.8 GB
- **Total data: ~2.2 GB**

### Option B: Full FLAIR Dataset (~100+ GB)

For production results with significantly more data. Downloads ~25-30 GB of compressed data, expands to 100+ GB when extracted.

Follow instructions in [data/README.md](data/README.md).

**Storage requirements:**
- `data/raw/` (full dataset): ~100+ GB
- `data/processed/` (processed dataset): ~50+ GB
- **Total data: ~150+ GB**

## How to Reproduce Results

### Quick Start (3 Steps in Colab)

1. **Download & explore data:**
   - Open `notebooks/00_download_toy_dataset.ipynb` in Colab → Run all cells
   - Open `notebooks/01_eda.ipynb` → Run all cells (generates class_map.json and splits)

2. **Train a model:**
   - Open one of these in Colab and run all cells:
     - `notebooks/02_train_resnet34_unet.ipynb`
     - `notebooks/03_train_convnext_tiny.ipynb`
     - `notebooks/04_train_dinov3_convnext_tiny.ipynb`

3. **Evaluate and reproduce:**
   - Open `notebooks/08_reproduce_results.ipynb` → Run all cells (runs all models and compares results)
   - Open `notebooks/06_test_metrics.ipynb` → Run all cells (detailed per-class metrics)

### Key Experiments

**Channel Comparison (3 vs 5 channels):**
- Open `notebooks/07_channel_comparison.ipynb` in Colab → Run all cells

Tests whether using all 5 FLAIR channels improves over using just RGB (channels 1-3).

**Conformal Prediction (Uncertainty Quantification):**
- Open `notebooks/05_conformal_prediction.ipynb` in Colab → Run all cells

Generates confidence intervals around predictions using calibration data.

### Training a Model from Config (Colab)

Train any model using the YAML config system in a Colab notebook cell:

```python
from src.train import train_from_config

train_from_config("configs/resnet34_unet_5ch.yaml")
```

Or create a new config and run:
```python
from src.train import train_from_config

# (Optionally edit the config first)
train_from_config("configs/my_experiment.yaml")
```

### Configuration Options

Key configuration parameters in YAML files:

```yaml
data:
  channels: [1, 2, 3, 4, 5]          # FLAIR bands to use
  batch_size: 8
  image_size: 512
  
model:
  name: convnext_tiny              # Model architecture
  pretrained: true
  in_channels: 5

training:
  max_epochs: 100
  optimizer:
    backbone_lr: 0.00001           # Lower LR for backbone
    head_lr: 0.0001                # Higher LR for segmentation head
  loss:
    name: weight_cross_entropy      # Handle class imbalance
```

## Supported Models

| Model | Config | Backbone | Pretrained | Params |
|-------|--------|----------|-----------|--------|
| ResNet34-UNet | `resnet34_unet_5ch.yaml` | ResNet34 | ImageNet | ~23M |
| ConvNeXt-Tiny | `convnext_tiny_5ch.yaml` | ConvNeXt-Tiny | ImageNet-21K | ~29M |
| DINOv3 + ConvNeXt-Tiny | `dinov3_convnext_tiny_5ch.yaml` | DINOv3 backbone | Self-supervised | ~23M |

All models support 3-channel (RGB) and 5-channel (all FLAIR bands) configurations.

## Monitoring Training

The code logs metrics to **Weights & Biases** by default. To disable W&B:

Edit the config file and set:
```yaml
wandb:
  enabled: false
```

Or set in code:
```python
os.environ["WANDB_DISABLED"] = "true"
```

## Expected Results

Key metrics on FLAIR toy dataset (validation set):

- **ResNet34-UNet:** ~0.75 mIoU (baseline)
- **ConvNeXt-Tiny:** ~0.78 mIoU (improved efficiency)
- **DINOv3-ConvNeXt-Tiny:** ~0.80 mIoU (best performance)

Results improve significantly when using 5 channels vs. 3 channels (+2-3% mIoU).

## Citation

If you use this code, please cite the FLAIR dataset:

```bibtex
@article{ign2022flair1,
  doi = {10.13140/RG.2.2.30183.73128/1},
  url = {https://arxiv.org/pdf/2211.12979.pdf},
  author = {Garioud, Anatol and others},
  title = {FLAIR \#1: semantic segmentation and domain adaptation dataset},
  year = {2022}
}
```

## Troubleshooting

**Out of Memory (OOM) in Colab:**
- Click **Runtime** → **Disconnect and delete all** → **Runtime** → **Change runtime type** → Select **A100** (if available)
- Reduce `batch_size` in config (try 4 or 2)
- Reduce `image_size` (try 256)

**Missing API Keys:**
- Ensure `WANDB_API_KEY` and `HF_TOKEN` are added to Colab Secrets (🔑 icon in left sidebar)
- To disable W&B: Add a cell with `import os; os.environ["WANDB_DISABLED"] = "true"` before running training

**Data Download Fails:**
- Check internet connection in Colab
- Verify Google Drive has enough free space (~2.2 GB for toy dataset)

**GPU Not Available in Colab:**
- Go to **Runtime** → **Change runtime type** → Ensure **GPU** is selected
- If T4 is unavailable, try A100, L4, or V100

## License

See [LICENSE](LICENSE) file.
