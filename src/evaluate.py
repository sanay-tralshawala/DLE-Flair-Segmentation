"""Validation and evaluation metrics for segmentation models."""
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics.classification import MulticlassJaccardIndex 
import numpy as np



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
    
def evaluate_conformal(model, loader, q_hat, num_classes, device):
    """
    Returns:
        coverage        — marginal pixel coverage
        avg_set_size    — mean number of classes in prediction set per pixel
        per_class_cov   — per-class conditional coverage  (num_classes,)
        set_size_maps   — list of (B, H, W) int tensors, one per batch
        pred_set_maps   — list of (B, C, H, W) bool tensors, one per batch
        all_probs       — list of (B, C, H, W) float tensors
        all_images      — list of (B, ch, H, W) float tensors
        all_masks       — list of (B, H, W) long tensors
    """
    threshold = 1.0 - q_hat

    total_pixels = covered_pixels = 0
    class_total   = np.zeros(num_classes)
    class_covered = np.zeros(num_classes)
    set_sizes_all = []

    set_size_maps = []
    pred_set_maps = []
    all_probs     = []
    all_images    = []
    all_masks     = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            imgs  = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(imgs)
            probs  = F.softmax(logits, dim=1)

            pred_sets = probs >= threshold              # (B, C, H, W) bool
            C = probs.shape[1]

            valid_gt = (masks >= 0) & (masks < C)      # (B, H, W)
            safe_masks = masks.clone()
            safe_masks[~valid_gt] = 0

            true_oh = F.one_hot(safe_masks, num_classes=C).permute(0, 3, 1, 2).bool()
            true_oh = true_oh & valid_gt.unsqueeze(1)  # zero out invalid pixels

            covered = (pred_sets & true_oh).any(dim=1) & valid_gt
            covered_pixels += covered.sum().item()
            total_pixels   += valid_gt.sum().item()

            for c in range(num_classes):
                c_mask = (masks == c)                  # pixels of true class c
                if c_mask.sum() == 0:
                    continue
                class_total[c]   += c_mask.sum().item()
                class_covered[c] += (pred_sets[:, c] & c_mask).sum().item()

            set_size = pred_sets.sum(dim=1)            # (B, H, W)
            set_sizes_all.append(set_size.cpu().numpy())

            set_size_maps.append(set_size.cpu())
            pred_set_maps.append(pred_sets.cpu())
            all_probs.append(probs.cpu())
            all_images.append(imgs.cpu())
            all_masks.append(masks.cpu())

    coverage    = covered_pixels / max(total_pixels, 1)
    avg_set_sz  = float(np.mean(np.concatenate(set_sizes_all)))
    per_class_cov = np.divide(
        class_covered, class_total,
        out=np.full(num_classes, float("nan")),
        where=class_total > 0
    )

    return coverage, avg_set_sz, per_class_cov, set_size_maps, pred_set_maps, all_probs, all_images, all_masks