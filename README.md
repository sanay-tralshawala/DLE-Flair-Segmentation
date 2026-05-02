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
Due to the storage requirement to store the model weights for this analysis please refer to this link to run the notebook based on the models stored on drive. Please mount this folder to your google drive and run the notebook located at `DLE-Flair-Segmentation/notebooks/08_reproduce_results.ipynb`
Link: https://drive.google.com/drive/folders/1l_PO7SDPt0AwFlYPfchzMQ_ondDVWBYU?usp=sharing


## How to retrain the models and generate new results?

1. **Run Exploratory Data Analysis:**
   - Open `notebooks/00_download_toy_dataset.ipynb` in Colab → Run all cells
   - Open `notebooks/01_eda.ipynb` → Run all cells (generates class_map.json and splits)

2. **Train a models:**
   - Open these notebooks in Colab and run all cells twice*:
     - `notebooks/02_train_resnet34_unet.ipynb`
     - `notebooks/03_train_convnext_tiny.ipynb`
     - `notebooks/04_train_dinov3_convnext_tiny.ipynb`
  * In Cell 6 you will find a .yaml file ending in "_3ch", this signifies that the 3 channel model will trained. To train the 5 channel model replace this part of the file name with "_5ch".
Note: wandb logging is currently turned off and controlled through the configuration yaml files.

3. **Evaluate and Analysis:**
   - Open `notebooks/05_Conformal_Prediction.ipynb` → Run all cells (Performs uncertainty study)
   - Open `notebooks/06_test_metrics.ipynb` → Run all cells (detailed per-class metrics)
   - Open `notebooks/07_channel_comparison.ipynb` in Colab → Run all cells (performs ablation study)
4. **Rerun reproduction**
   - At this point, since all models are regenerated:
   -   Open `notebooks/08_reproduce_results.ipynb` → Run all cells (Reproduce results in a condensed format)
## License

See [LICENSE](LICENSE) file.
