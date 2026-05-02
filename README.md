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
This codebase is optimized for a Google Colab enviornment, however cloning this repository locally and running through this procedure in a local enviornment (excluding mmounting google drive) would work as well.

**Prerequisites:**
- Google account with access to Google Colab 
- T4 GPU generally is acceptable but you may run into runtime errors during the training of the DINOv3 models where an A100 GPU is sufficient (requires Colab pro access)

### Setup Instructions

1. **Upload repository to Colab:**
   - Mount your Google Drive in Colab
   -   In the desired holding location, open a terminal or a Colab Jupyter Notebook and run the following commands:
```python
from google.colab import drive
drive.mount('/content/drive')
```
Next, change the directory to the location where you would like to clone the repository. The code is currently built for the repo cloning to occur within a folder named "Deep Learning Project" with MyDrive, while this is not required, the file path must be changed accordingly in future notebooks.
```python
%cd /content/drive/MyDrive/{path to desired directory}
```

Clone the repository to this specified directory by the command:
```python
!git clone https://github.com/sanay-tralshawala/DLE-Flair-Segmentation/
```
   
2. **Gain access to required models on Hugging Face**
   - Log in or create an account to HuggingFace
   - Go to a DINOv3 model (https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m) and request access to their family of models which are gated
   - You will not be able to train or reproduce results for the DINOv3 model analysis unless you have gained access to this model

3. **Add Colab secrets for API keys:**
   - Click the 🔑 **Secrets** icon in the left sidebar
   - Add `HF_TOKEN` — For Hugging Face gated models like DINOv3 (get from [huggingface.co](https://huggingface.co/settings/tokens))
   - The notebooks will automatically load these via `userdata.get_secret()`
   - Ensure the notebook has been granted access to the token

4. **Enable GPU (required for training):**
   - Go to **Runtime** → **Change runtime type** → Select **GPU** (T4 or A100 recommended)

## How to Reproduce Results

If you would like to simply reproduce the existing results, refer to `notebooks/08_reproduce_results.ipynb`



<!--
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
