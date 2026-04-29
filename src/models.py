"""Model construction and backbone freezing helpers."""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import segmentation_models_pytorch as smp
from transformers import AutoModel

class SegmentationModel(nn.Module):

    def __init__(self, backbone_name: str, in_channels: int = 5, num_classes: int = 5):
        super().__init__()

        if backbone_name not in timm.list_models():
            raise ValueError(f"Backbone '{backbone_name}' not found in timm model zoo.")

        # Backbone (feature extractor)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            in_chans=in_channels,
            features_only=True
        )

        # Get channels of deepest feature map
        encoder_channels = self.backbone.feature_info.channels()
        in_ch = encoder_channels[-1]

        # Simple segmentation head
        self.decoder = nn.Sequential(
            nn.Conv2d(in_ch, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]  # (H, W)

        features = self.backbone(x)
        x = features[-1]  # deepest feature map

        x = self.decoder(x)

        # Upsample to original resolution
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)

        return x

class DinoV3SegmentationModel(nn.Module):
    def __init__(self, model_name="facebook/dinov3-convnext-tiny-pretrain-lvd1689m", in_channels=5, num_classes=5):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(model_name, token=os.environ.get("HF_HUB_TOKEN"))

        # Stage 0: downsample_layers[0] is the stem Conv2d (3 -> 96, 4x4 stride 4)
        stem_conv = self.backbone.stages[0].downsample_layers[0]  # no .model
        if in_channels != stem_conv.in_channels:
            new_stem_conv = nn.Conv2d(
                in_channels,
                stem_conv.out_channels,
                kernel_size=stem_conv.kernel_size,
                stride=stem_conv.stride,
                padding=stem_conv.padding,
                bias=stem_conv.bias is not None,
            )
            with torch.no_grad():
                copied_channels = min(in_channels, stem_conv.in_channels)
                new_stem_conv.weight[:, :copied_channels].copy_(stem_conv.weight[:, :copied_channels])
                if in_channels > stem_conv.in_channels:
                    repeated = stem_conv.weight.mean(dim=1, keepdim=True)
                    for ch in range(stem_conv.in_channels, in_channels):
                        new_stem_conv.weight[:, ch:ch + 1].copy_(repeated)
                if stem_conv.bias is not None:
                    new_stem_conv.bias.copy_(stem_conv.bias)
            self.backbone.stages[0].downsample_layers[0] = new_stem_conv  # no .model

        # Deepest stage outputs 768 channels
        hidden_dim = self.backbone.config.hidden_sizes[-1]  # 768

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.shape[-2:]

        features = x
        for stage in self.backbone.stages:
            features = stage(features)

        x = self.decoder(features)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x

def build_model(model_config: dict):
    """Return a segmentation model using the configured input channels and class count."""
    in_channels = model_config.get("in_channels", 5)
    num_classes = model_config["num_classes"]

    if model_config["name"] == "resnet34_unet":
        model = smp.Unet(
            encoder_name=model_config.get("encoder", "resnet34"),
            encoder_weights="imagenet" if model_config.get("pretrained", True) else None,
            in_channels=in_channels,
            classes=num_classes,
        )
    elif model_config["name"] == "resnet34":
        model = SegmentationModel(
            backbone_name=model_config.get("encoder", "resnet34"),
            in_channels=in_channels,
            num_classes=num_classes,
        )
    elif model_config["name"] == "convnext_tiny":
        model = SegmentationModel(
            backbone_name=model_config.get("encoder", "convnext_tiny"),
            in_channels=in_channels,
            num_classes=num_classes,
        )
    elif model_config["name"] == "dinov3_convnext_tiny":
        model = DinoV3SegmentationModel(in_channels=in_channels, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model name: {model_config['name']}")

    return model


def freeze_backbone(model) -> None:
    """Disable gradients for the model backbone during head warmup."""
    if hasattr(model, 'encoder'):
        for p in model.encoder.parameters():
            p.requires_grad = False
    elif hasattr(model, 'backbone'):
        for p in model.backbone.parameters():
            p.requires_grad = False
    else:
        print("Warning: Model has neither 'encoder' nor 'backbone' attribute to freeze.")

def unfreeze_backbone(model) -> None:
    """Enable gradients for the model backbone after head warmup."""
    if hasattr(model, 'encoder'):
        for p in model.encoder.parameters():
            p.requires_grad = True
    elif hasattr(model, 'backbone'):
        for p in model.backbone.parameters():
            p.requires_grad = True
