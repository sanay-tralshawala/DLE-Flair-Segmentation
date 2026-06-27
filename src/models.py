"""Model construction and backbone freezing helpers."""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import segmentation_models_pytorch as smp
from transformers import AutoModel


class SegmentationModel(nn.Module):

    def __init__(
        self,
        backbone_name: str,
        in_channels: int = 5,
        num_classes: int = 5,
        pretrained: bool = True,
    ):
        super().__init__()

        if backbone_name not in timm.list_models():
            raise ValueError(f"Backbone '{backbone_name}' not found in timm model zoo.")

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
        )

        encoder_channels = self.backbone.feature_info.channels()
        in_ch = encoder_channels[-1]

        self.decoder = nn.Sequential(
            nn.Conv2d(in_ch, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        features = self.backbone(x)
        x = features[-1]
        x = self.decoder(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x


class DinoV3SegmentationModel(nn.Module):

    def __init__(
        self,
        model_name: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        in_channels: int = 5,
        num_classes: int = 5,
    ):
        super().__init__()

        hf_model = AutoModel.from_pretrained(
            model_name, token=os.environ.get("HF_HUB_TOKEN")
        )

        # Always unwrap .model if present so self.backbone owns .stages directly.
        # This ensures state dict keys are always backbone.stages.* regardless
        # of transformers version.
        self.backbone = getattr(hf_model, "model", hf_model)

        stem_conv = self.backbone.stages[0].downsample_layers[0]
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
                new_stem_conv.weight[:, :copied_channels].copy_(
                    stem_conv.weight[:, :copied_channels]
                )
                if in_channels > stem_conv.in_channels:
                    repeated = stem_conv.weight.mean(dim=1, keepdim=True)
                    for ch in range(stem_conv.in_channels, in_channels):
                        new_stem_conv.weight[:, ch : ch + 1].copy_(repeated)
                if stem_conv.bias is not None:
                    new_stem_conv.bias.copy_(stem_conv.bias)
            self.backbone.stages[0].downsample_layers[0] = new_stem_conv

        hidden_dim = self.backbone.config.hidden_sizes[-1]  # 768 for tiny

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Backwards compatibility: checkpoints saved when the HF model was
        # stored under self.backbone.model have keys like backbone.model.stages.*
        # Remap them to backbone.stages.* transparently so every caller can use
        # the standard model.load_state_dict(checkpoint["model_state_dict"]) —
        # no special helper needed anywhere in the codebase.
        if any(k.startswith("backbone.model.") for k in state_dict):
            state_dict = {
                k.replace("backbone.model.", "backbone."): v
                for k, v in state_dict.items()
            }
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        features = x
        for stage in self.backbone.stages:
            features = stage(features)

        x = self.decoder(features)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x


def build_model(model_config: dict) -> nn.Module:
    """Return a segmentation model using the configured input channels and class count."""
    in_channels = model_config.get("in_channels", 5)
    num_classes = model_config["num_classes"]

    if model_config["name"] == "resnet34_unet":
        return smp.Unet(
            encoder_name=model_config.get("encoder", "resnet34"),
            encoder_weights="imagenet" if model_config.get("pretrained", True) else None,
            in_channels=in_channels,
            classes=num_classes,
        )

    if model_config["name"] == "resnet34":
        return SegmentationModel(
            backbone_name=model_config.get("encoder", "resnet34"),
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained=model_config.get("pretrained", True),
        )

    if model_config["name"] == "convnext_tiny":
        return SegmentationModel(
            backbone_name=model_config.get("encoder", "convnext_tiny"),
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained=model_config.get("pretrained", True),
        )

    if model_config["name"] == "dinov3_convnext_tiny":
        return DinoV3SegmentationModel(
            in_channels=in_channels,
            num_classes=num_classes,
        )

    raise ValueError(f"Unsupported model name: {model_config['name']}")


def freeze_backbone(model: nn.Module) -> None:
    """Disable gradients for the model backbone during head warmup."""
    if hasattr(model, "encoder"):
        for p in model.encoder.parameters():
            p.requires_grad = False
    elif hasattr(model, "backbone"):
        for p in model.backbone.parameters():
            p.requires_grad = False
    else:
        print("Warning: Model has neither 'encoder' nor 'backbone' attribute to freeze.")


def unfreeze_backbone(model: nn.Module) -> None:
    """Enable gradients for the model backbone after head warmup."""
    if hasattr(model, "encoder"):
        for p in model.encoder.parameters():
            p.requires_grad = True
    elif hasattr(model, "backbone"):
        for p in model.backbone.parameters():
            p.requires_grad = True