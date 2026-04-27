"""Model construction and backbone freezing helpers."""
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
    def __init__(self, model_name="facebook/dinov3-convnext-tiny-pretrain-lvd1689m", num_classes=5):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(model_name)

        # hidden size depends on model (tiny ≈ 768)
        hidden_dim = self.backbone.config.hidden_sizes[-1]

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.shape[-2:]

        outputs = self.backbone(x)

        # HF convnext returns feature maps already
        if hasattr(outputs, "last_hidden_state"):
            x = outputs.last_hidden_state
        elif hasattr(outputs, "hidden_states"):
            x = outputs.hidden_states[-1]
        else:
            raise ValueError("Unexpected DINO output format")

        x = self.decoder(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)

        return x

def build_model(model_config: dict):
    """Return a segmentation model with 5 input channels, backbone/encoder, and 5-class head."""
    if model_config["name"] == "resnet34_unet":
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=5,
            classes=5
        )
    elif model_config["name"] == "resnet34":
        model = SegmentationModel(backbone_name="resnet34", in_channels=5, num_classes=5)
    elif model_config["name"] == "convnext_tiny":
        model = SegmentationModel(backbone_name="convnext_tiny", in_channels=5, num_classes=5)
    elif model_config["name"] == "dinov3_convnext_tiny":
        model = DinoV3SegmentationModel(num_classes=5)
    else:
        raise ValueError(f"Unsupported model name: {model_config['name']}")

    return model

def freeze_backbone(model) -> None:
    """Disable gradients for the model backbone during head warmup."""
    for p in model.backbone.parameters():
        p.requires_grad = False


def unfreeze_backbone(model) -> None:
    """Enable gradients for the model backbone after head warmup."""
    for p in model.backbone.parameters():
        p.requires_grad = True
