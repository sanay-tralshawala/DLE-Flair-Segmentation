'''
Shared training loop for FLAIR segmentation models.

Primary entry point: `train_from_config(<model_config_path>)`
'''

from pathlib import Path
import json
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm #  progress bars
import wandb # experiment tracking

# (future) local imports
from src.data import build_dataloaders
from src.models import build_model, freeze_backbone, unfreeze_backbone
from src.evaluate import evaluate_model
from src.utils import load_config, get_device



def build_loss(NotImplemented) -> nn.Module:
    '''
    Build the segmentation loss function.
    
    Two options:
        - "cross_entropy": standard cross-entropy loss for multi-class segmentation
        - "weighted_cross_entropy": cross-entropy loss with class weights to handle class imbalance

    Expected logits: [batch_size, num_classes, height, width]
        Expected targets: [batch_size, height, width] with class indices in [0, num_classes-1]
    '''


def build_optimizer(NotImplemented) -> optim.Optimizer:
    '''
    Build the optimizer for training.
    
    Two options:
        - "adam": Adam optimizer with default parameters
        - "adamw": AdamW optimizer with default parameters
    '''


def train_one_epoch(NotImplemented) -> dict:
    '''
    Train the model for one epoch and return the average training loss.
    '''

@torch.no_grad()
def validate_one_epoch(NotImplemented) -> dict:
    '''
    Validate the model for one epoch and return the average validation loss and metrics.
    '''

def save_checkpoint(NotImplemented) -> None:
    '''
    Save the model checkpoint to local disk. 
    '''

def train_from_config(config_path: Path) -> None:
    '''
    Main training loop that trains a FLAIR segmentation model based on the provided configuration file.
    '''

