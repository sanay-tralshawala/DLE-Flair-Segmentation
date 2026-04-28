"""Validation and evaluation metrics for segmentation models."""
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics.classification import MulticlassJaccardIndex 


def _unpack_batch(batch):
    if isinstance(batch, dict):
        return batch["image"], batch["mask"]
    return batch


def _class_name(class_names, class_id: int) -> str:
    if class_names is None:
        return str(class_id)
    if isinstance(class_names, dict):
        return str(class_names.get(class_id, class_names.get(str(class_id), class_id)))
    if 0 <= class_id < len(class_names):
        return str(class_names[class_id])
    return str(class_id)


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


def evaluate_class_iou(model, dataloader, num_classes, device, ignore_index=None, class_names=None) -> dict:
    """Return per-class IoU metrics for a segmentation model."""
    model.eval()
    iou_metric = MulticlassJaccardIndex(
        num_classes=num_classes,
        ignore_index=ignore_index,
        average=None,
    ).to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating class IoU"):
            images, masks = _unpack_batch(batch)
            images = images.to(device)
            masks = masks.to(device).long()

            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            iou_metric.update(predictions, masks)

    iou_values = iou_metric.compute().detach().cpu().tolist()
    class_iou = {
        class_id: float(iou_values[class_id])
        for class_id in range(num_classes)
    }
    class_iou_rows = [
        {
            "class_id": class_id,
            "class_name": _class_name(class_names, class_id),
            "iou": class_iou[class_id],
        }
        for class_id in range(num_classes)
    ]
    return {"class_iou": class_iou, "class_iou_rows": class_iou_rows}
