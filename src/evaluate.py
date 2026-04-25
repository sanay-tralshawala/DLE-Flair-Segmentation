"""Validation and evaluation metrics for segmentation models."""


def evaluate_model(model, dataloader, loss_fn, device) -> dict:
    """Return validation metrics with at least `val_loss`; add IoU here later."""
    raise NotImplementedError
