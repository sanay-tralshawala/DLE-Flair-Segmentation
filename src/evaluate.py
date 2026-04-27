"""Validation and evaluation metrics for segmentation models."""
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torchmetrics.classification import MulticlassJaccardIndex 


def evaluate_model(model, dataloader, num_classes, loss_fn, device, ignore_index=None) -> dict:
    """Return validation metrics with at least `val_loss`and Intersection over Union"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    iou_metric = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=ignore_index, average='macro').to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"): # Using tqdm for progress bar
            images, masks = batch["image"], batch["mask"]
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = loss_fn(logits, masks)
            total_loss += loss.item()
            num_batches += 1
            
            predictions = torch.argmax(logits, dim=1)
            iou_metric.update(predictions, masks)
            
    mean_iou = iou_metric.compute().item()

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return {"val_loss": avg_loss, "val_IoU": mean_iou}

