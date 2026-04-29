"""Validation and evaluation metrics for segmentation models."""
import time

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


def load_checkpoint_model(checkpoint_path, device, model_config: dict | None = None):
    """Build a model from config, load checkpoint weights, and switch to eval mode."""
    from src.models import build_model

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    model_config = model_config or config.get("model")
    if model_config is None:
        raise ValueError("A model config is required when the checkpoint has no embedded config.")

    model_config = dict(model_config)
    if model_config.get("name") != "dinov3_convnext_tiny":
        model_config["pretrained"] = False

    model = build_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def confusion_matrix_from_tensors(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> torch.Tensor:
    """Return a [true_class, predicted_class] confusion matrix for one batch."""
    predictions = predictions.reshape(-1).long()
    targets = targets.reshape(-1).long()
    valid = (targets >= 0) & (targets < num_classes)
    if ignore_index is not None:
        valid = valid & (targets != ignore_index)

    predictions = predictions[valid]
    targets = targets[valid]
    if targets.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.long, device=targets.device)

    bins = targets * num_classes + predictions
    matrix = torch.bincount(bins, minlength=num_classes * num_classes)
    return matrix.reshape(num_classes, num_classes)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=denominator != 0,
    )


def _safe_nanmean(values: np.ndarray) -> float:
    return float(np.nanmean(values)) if not np.all(np.isnan(values)) else float("nan")


def summarize_confusion_matrix(confusion_matrix, class_names=None) -> dict:
    """Compute accuracy, IoU, and Dice metrics from a confusion matrix."""
    matrix = np.asarray(confusion_matrix, dtype=np.float64)
    true_pixels = matrix.sum(axis=1)
    predicted_pixels = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    total = matrix.sum()

    class_iou = _safe_divide(true_positive, true_pixels + predicted_pixels - true_positive)
    class_dice = _safe_divide(2 * true_positive, true_pixels + predicted_pixels)
    pixel_accuracy = float(true_positive.sum() / total) if total else float("nan")

    class_rows = [
        {
            "class_id": class_id,
            "class_name": _class_name(class_names, class_id),
            "iou": float(class_iou[class_id]),
            "dice": float(class_dice[class_id]),
            "support_pixels": int(true_pixels[class_id]),
            "predicted_pixels": int(predicted_pixels[class_id]),
        }
        for class_id in range(matrix.shape[0])
    ]

    return {
        "pixel_accuracy": pixel_accuracy,
        "mIoU": _safe_nanmean(class_iou),
        "mean_dice": _safe_nanmean(class_dice),
        "class_iou": class_iou,
        "class_dice": class_dice,
        "class_rows": class_rows,
    }


@torch.no_grad()
def evaluate_test_metrics(
    model,
    dataloader,
    num_classes: int,
    device,
    ignore_index: int | None = 255,
    class_names=None,
    input_transform=None,
    desc: str = "Evaluating test metrics",
    include_average_precision: bool = False,
    latency_warmup_batches: int = 1,
) -> dict:
    """Evaluate semantic segmentation metrics on a dataloader."""
    model.eval()
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)
    average_precision = None
    if include_average_precision:
        from torchmetrics.classification import MulticlassAveragePrecision

        average_precision = MulticlassAveragePrecision(num_classes=num_classes, average=None).to(device)
    valid_pixel_count = 0
    latency_warmup_batches = max(0, int(latency_warmup_batches))
    latency_total_seconds = 0.0
    latency_num_patches = 0

    for batch_index, batch in enumerate(tqdm(dataloader, desc=desc)):
        images, masks = _unpack_batch(batch)
        if input_transform is not None:
            images = input_transform(images)
        images = images.to(device)
        masks = masks.to(device).long()

        should_time_batch = batch_index >= latency_warmup_batches
        if should_time_batch and images.is_cuda:
            torch.cuda.synchronize(images.device)
        start_time = time.perf_counter() if should_time_batch else None
        logits = model(images)
        if should_time_batch:
            if images.is_cuda:
                torch.cuda.synchronize(images.device)
            latency_total_seconds += time.perf_counter() - start_time
            latency_num_patches += int(images.shape[0])

        probabilities = F.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

        confusion_matrix += confusion_matrix_from_tensors(
            predictions=predictions,
            targets=masks,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

        flat_masks = masks.reshape(-1)
        valid = (flat_masks >= 0) & (flat_masks < num_classes)
        if ignore_index is not None:
            valid = valid & (flat_masks != ignore_index)
        if valid.any():
            valid_pixel_count += int(valid.sum().item())
            if average_precision is not None:
                flat_probs = probabilities.permute(0, 2, 3, 1).reshape(-1, num_classes)
                average_precision.update(flat_probs[valid], flat_masks[valid])

    confusion_np = confusion_matrix.detach().cpu().numpy()
    summary = summarize_confusion_matrix(confusion_np, class_names=class_names)
    metrics = {
        "confusion_matrix": confusion_np,
        "pixel_accuracy": summary["pixel_accuracy"],
        "mIoU": summary["mIoU"],
        "mean_dice": summary["mean_dice"],
        "class_iou": summary["class_iou"],
        "class_dice": summary["class_dice"],
        "class_rows": summary["class_rows"],
        "num_pixels": int(valid_pixel_count),
        "latency_ms_per_patch": (
            latency_total_seconds * 1000.0 / latency_num_patches
            if latency_num_patches
            else float("nan")
        ),
        "latency_total_ms": latency_total_seconds * 1000.0,
        "latency_num_patches": latency_num_patches,
    }
    if average_precision is not None:
        if valid_pixel_count:
            class_ap = average_precision.compute().detach().cpu().numpy()
        else:
            class_ap = np.full(num_classes, np.nan, dtype=np.float64)
        metrics["mAP"] = _safe_nanmean(class_ap)
        metrics["class_ap"] = class_ap
    return metrics
    
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
