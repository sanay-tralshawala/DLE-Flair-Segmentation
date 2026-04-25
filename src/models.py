"""Model construction and backbone freezing helpers."""


def build_model(model_config: dict):
    """Return a segmentation model with a `.backbone` or `.encoder` and 5-class head."""
    raise NotImplementedError


def freeze_backbone(model) -> None:
    """Disable gradients for the model backbone during head warmup."""
    raise NotImplementedError


def unfreeze_backbone(model) -> None:
    """Enable gradients for the model backbone after head warmup."""
    raise NotImplementedError
