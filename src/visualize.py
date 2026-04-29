"""Visualization helpers for segmentation masks and prediction logits."""

from __future__ import annotations

import colorsys
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches




def _to_numpy(value) -> np.ndarray:
    """Convert torch/numpy-like tensors to a CPU numpy array."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _class_name(class_names, label: int) -> str:
    if class_names is None:
        return str(label)
    if isinstance(class_names, dict):
        return str(class_names.get(label, class_names.get(str(label), label)))
    if isinstance(class_names, Sequence) and not isinstance(class_names, str):
        if 0 <= label < len(class_names):
            return str(class_names[label])
    return str(label)


def _palette_color(label: int, palette: dict | Sequence | None = None) -> tuple[int, int, int]:
    if palette is not None:
        if isinstance(palette, dict):
            color = palette.get(label, palette.get(str(label)))
        else:
            color = palette[label] if 0 <= label < len(palette) else None
        if color is not None:
            arr = np.asarray(color, dtype=float)
            if arr.max(initial=0) <= 1.0:
                arr = arr * 255
            return tuple(arr[:3].astype(np.uint8).tolist())

    hue = (int(label) * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return int(red * 255), int(green * 255), int(blue * 255)


def _single_mask(mask) -> np.ndarray:
    mask = _to_numpy(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected a single mask shaped [H, W], got shape {mask.shape}.")
    return mask.astype(np.int64, copy=False)


def _mask_batch(masks) -> np.ndarray:
    masks = _to_numpy(masks)
    if masks.ndim == 2:
        return masks[None, ...].astype(np.int64, copy=False)
    if masks.ndim == 3:
        return masks.astype(np.int64, copy=False)
    raise ValueError(f"Expected masks shaped [H, W] or [B, H, W], got shape {masks.shape}.")


def _single_image(image, channels: tuple[int, int, int] = (0, 1, 2), percent_clip: tuple[int, int] = (2, 98)) -> np.ndarray:
    image = _to_numpy(image).astype(np.float32, copy=False)
    if image.ndim != 3:
        raise ValueError(f"Expected a single image shaped [C, H, W] or [H, W, C], got shape {image.shape}.")

    if image.shape[-1] <= 10 and image.shape[0] > 10:
        image = np.moveaxis(image, -1, 0)

    rgb = image[list(channels)]
    rgb = np.moveaxis(rgb, 0, -1)
    low, high = np.percentile(rgb, percent_clip)
    if high <= low:
        return np.zeros_like(rgb, dtype=np.float32)
    return np.clip((rgb - low) / (high - low), 0, 1)


def _image_batch(images) -> np.ndarray:
    images = _to_numpy(images)
    if images.ndim == 3:
        return images[None, ...]
    if images.ndim == 4:
        return images
    raise ValueError(f"Expected images shaped [C, H, W], [H, W, C], [B, C, H, W], or [B, H, W, C], got shape {images.shape}.")


def _add_mask_legend(axis, mask: np.ndarray, class_names=None, palette=None, ignore_index: int = 255) -> None:
    labels = [int(label) for label in np.unique(mask)]
    handles = []
    for label in labels:
        if label == ignore_index:
            color = np.array([0.35, 0.35, 0.35])
            name = "ignore"
        else:
            color = np.array(_palette_color(label, palette), dtype=float) / 255.0
            name = _class_name(class_names, label)
        handles.append(Patch(facecolor=color, edgecolor="none", label=name))
    if handles:
        axis.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            ncol=1,
            borderaxespad=0.0,
            fontsize="small",
        )


def colorize_label_mask(mask, class_names=None, palette=None, ignore_index: int = 255) -> np.ndarray:
    """Return an RGB visualization for a label mask shaped [H, W]."""
    del class_names  # Class names are used by plotting legends, not raw RGB arrays.
    mask = _single_mask(mask)
    colorized = np.zeros((*mask.shape, 3), dtype=np.uint8)

    for label in np.unique(mask):
        label = int(label)
        if label == ignore_index:
            colorized[mask == label] = (90, 90, 90)
        else:
            colorized[mask == label] = _palette_color(label, palette)

    return colorized


def overlay_mask(
    image,
    mask,
    alpha: float = 0.45,
    channels: tuple[int, int, int] = (0, 1, 2),
    class_names=None,
    palette=None,
    ignore_index: int = 255,
) -> np.ndarray:
    """Overlay a label mask on one image and return an RGB array."""
    del class_names
    rgb = _single_image(image, channels=channels)
    mask = _single_mask(mask)
    color_mask = colorize_label_mask(mask, palette=palette, ignore_index=ignore_index).astype(np.float32) / 255.0
    visible = mask != ignore_index
    overlay = rgb.copy()
    overlay[visible] = (1 - alpha) * overlay[visible] + alpha * color_mask[visible]
    return np.clip(overlay, 0, 1)


def plot_mask(mask, title: str | None = None, class_names=None, palette=None, ignore_index: int = 255):
    """Plot one ground-truth label mask and return the Matplotlib figure."""
    mask = _single_mask(mask)
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.imshow(colorize_label_mask(mask, palette=palette, ignore_index=ignore_index))
    axis.set_title(title or "Mask")
    axis.axis("off")
    _add_mask_legend(axis, mask, class_names=class_names, palette=palette, ignore_index=ignore_index)
    fig.tight_layout()
    return fig


def plot_overlay_grid(
    images,
    masks,
    max_items: int = 4,
    alpha: float = 0.45,
    channels: tuple[int, int, int] = (0, 1, 2),
    title: str = "Overlay",
    class_names=None,
    palette=None,
    ignore_index: int = 255,
):
    """Plot image/mask overlays for matching image and mask batches."""
    images = _image_batch(images)
    masks = _mask_batch(masks)
    count = min(max_items, len(images), len(masks))
    if count < 1:
        raise ValueError("No image/mask pairs available to plot.")

    fig, axes = plt.subplots(1, count, figsize=(5 * count, 5), squeeze=False)
    for index, axis in enumerate(axes[0]):
        mask = masks[index]
        axis.imshow(
            overlay_mask(
                images[index],
                mask,
                alpha=alpha,
                channels=channels,
                palette=palette,
                ignore_index=ignore_index,
            )
        )
        axis.set_title(f"{title} {index}")
        axis.axis("off")
        _add_mask_legend(axis, mask, class_names=class_names, palette=palette, ignore_index=ignore_index)

    fig.tight_layout()
    return fig


def plot_mask_grid(masks, max_items: int = 4, class_names=None, palette=None, ignore_index: int = 255):
    """Plot a small grid of ground-truth masks shaped [B, H, W] or [H, W]."""
    masks = _mask_batch(masks)
    count = min(max_items, len(masks))
    if count < 1:
        raise ValueError("No masks available to plot.")

    fig, axes = plt.subplots(1, count, figsize=(5 * count, 5), squeeze=False)
    for index, axis in enumerate(axes[0]):
        mask = masks[index]
        axis.imshow(colorize_label_mask(mask, palette=palette, ignore_index=ignore_index))
        axis.set_title(f"Mask {index}")
        axis.axis("off")
        _add_mask_legend(axis, mask, class_names=class_names, palette=palette, ignore_index=ignore_index)

    fig.tight_layout()
    return fig


def logits_to_label_mask(logits) -> np.ndarray:
    """Convert logits shaped [K, H, W] or [B, K, H, W] to label masks."""
    logits = _to_numpy(logits)
    if logits.ndim == 3:
        return np.argmax(logits, axis=0).astype(np.int64)
    if logits.ndim == 4:
        return np.argmax(logits, axis=1).astype(np.int64)
    raise ValueError(f"Expected logits shaped [K, H, W] or [B, K, H, W], got shape {logits.shape}.")


def logits_to_color_image(logits, class_names=None, palette=None, ignore_index: int = 255) -> np.ndarray:
    """Convert logits shaped [K, H, W] or [B, K, H, W] to RGB prediction images."""
    mask = logits_to_label_mask(logits)
    if mask.ndim == 2:
        return colorize_label_mask(mask, class_names=class_names, palette=palette, ignore_index=ignore_index)
    return np.stack([
        colorize_label_mask(item, class_names=class_names, palette=palette, ignore_index=ignore_index)
        for item in mask
    ])


def plot_logits(
    logits,
    title: str | None = None,
    class_names=None,
    palette=None,
    ignore_index: int = 255,
    max_items: int = 4,
):
    """Plot prediction logits after argmax conversion and return the Matplotlib figure."""
    masks = _mask_batch(logits_to_label_mask(logits))
    count = min(max_items, len(masks))
    if count < 1:
        raise ValueError("No logits available to plot.")

    fig, axes = plt.subplots(1, count, figsize=(5 * count, 5), squeeze=False)
    for index, axis in enumerate(axes[0]):
        mask = masks[index]
        axis.imshow(colorize_label_mask(mask, palette=palette, ignore_index=ignore_index))
        axis.set_title(title or f"Prediction {index}")
        axis.axis("off")
        _add_mask_legend(axis, mask, class_names=class_names, palette=palette, ignore_index=ignore_index)

    fig.tight_layout()
    return fig


def plot_logits_overlay_grid(
    images,
    logits,
    max_items: int = 4,
    alpha: float = 0.45,
    channels: tuple[int, int, int] = (0, 1, 2),
    title: str = "Prediction",
    class_names=None,
    palette=None,
    ignore_index: int = 255,
):
    """Plot image/prediction overlays from matching image batches and logits."""
    return plot_overlay_grid(
        images,
        logits_to_label_mask(logits),
        max_items=max_items,
        alpha=alpha,
        channels=channels,
        title=title,
        class_names=class_names,
        palette=palette,
        ignore_index=ignore_index,
    )


def plot_prediction_comparison_grid(
    images,
    masks,
    logits,
    max_items: int = 4,
    alpha: float = 0.45,
    channels: tuple[int, int, int] = (0, 1, 2),
    class_names=None,
    palette=None,
    ignore_index: int = 255,
):
    """Plot ground-truth and prediction overlays for the same image batch."""
    images = _image_batch(images)
    masks = _mask_batch(masks)
    predictions = _mask_batch(logits_to_label_mask(logits))
    count = min(max_items, len(images), len(masks), len(predictions))
    if count < 1:
        raise ValueError("No image/mask/logit triples available to plot.")

    fig, axes = plt.subplots(count, 2, figsize=(10, 5 * count), squeeze=False)
    for index in range(count):
        columns = (
            ("Ground truth", masks[index]),
            ("Prediction", predictions[index]),
        )
        for column_index, (title, mask) in enumerate(columns):
            axis = axes[index][column_index]
            axis.imshow(
                overlay_mask(
                    images[index],
                    mask,
                    alpha=alpha,
                    channels=channels,
                    palette=palette,
                    ignore_index=ignore_index,
                )
            )
            axis.set_title(f"{title} {index}")
            axis.axis("off")
            _add_mask_legend(axis, mask, class_names=class_names, palette=palette, ignore_index=ignore_index)

    fig.tight_layout()
    return fig


def plot_class_iou(class_iou_result: dict, title: str = "Best checkpoint class IoU"):
    """Plot per-class IoU values returned by `evaluate_class_iou`."""
    rows = class_iou_result["class_iou_rows"]
    if not rows:
        raise ValueError("No class IoU rows available to plot.")

    labels = [f"{row['class_id']}: {row['class_name']}" for row in rows]
    values = [float(row["iou"]) for row in rows]

    fig_height = max(3, 0.55 * len(rows) + 1.5)
    fig, axis = plt.subplots(figsize=(8, fig_height))
    y_positions = np.arange(len(rows))
    bars = axis.barh(y_positions, values, color="#4C78A8")

    axis.set_yticks(y_positions)
    axis.set_yticklabels(labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.set_xlabel("IoU")
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.25)

    for value, bar in zip(values, bars):
        label_x = min(value + 0.02, 0.98)
        axis.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left" if value <= 0.95 else "right",
        )

    fig.tight_layout()
    return fig

#Conformal Prediction

def plot_score_and_coverage_summary(scores, q_hat, alpha, per_class_cov, class_names, model_label, viz_out_dir, model_key, dpi):
    """
    Plots the calibration score distribution and per-class conditional coverage.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Left: score histogram
    ax = axes[0]
    ax.hist(scores, bins=80, color="steelblue", edgecolor="none", alpha=0.85)
    ax.axvline(q_hat, color="crimson", lw=2, label=f"q̂ = {q_hat:.3f}")
    ax.set_xlabel("Nonconformity score  (1 − p_true)")
    ax.set_ylabel("Pixel count")
    ax.set_title("Calibration score distribution")
    ax.legend()

    # Right: per-class coverage bar chart
    ax = axes[1]
    colors = ["#2ecc71" if c >= 1 - alpha else "#e74c3c" for c in per_class_cov]
    bars = ax.barh(class_names, per_class_cov, color=colors)
    ax.axvline(1 - alpha, color="black", lw=1.5, ls="--", label=f"Target {1 - alpha:.0%}")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Coverage")
    ax.set_title("Per-class conditional coverage")
    ax.legend()
    for bar, val in zip(bars, per_class_cov):
        ax.text(min(val + 0.01, 1.0), bar.get_y() + bar.get_height()/2,
                f"{val:.1%}", va="center", fontsize=9)

    fig.suptitle(f"{model_label}  |  α={alpha}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    
    (viz_out_dir / f"{model_key}").mkdir(parents=True, exist_ok=True)
    fig.savefig(viz_out_dir / f"{model_key}" / f"{model_key}_coverage_summary.png", dpi=dpi)
    return fig
    
def plot_set_size_distribution(all_set_sizes, num_classes, avg_set_size, viz_out_dir, model_key, dpi):
    """
    Plots the distribution of prediction set sizes.
    """
    counts = np.bincount(all_set_sizes.astype(int), minlength=num_classes + 1)
    fracs  = counts / counts.sum()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(num_classes + 1), fracs[:num_classes + 1], color="steelblue", edgecolor="white")
    ax.set_xticks(range(num_classes + 1))
    ax.set_xlabel("Prediction set size")
    ax.set_ylabel("Fraction of pixels")
    ax.set_title(f"Set size distribution  (avg = {avg_set_size:.2f})")
    ax.axvline(avg_set_size, color="crimson", lw=1.5, ls="--", label=f"mean = {avg_set_size:.2f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(viz_out_dir / f"{model_key}"/ f"{model_key}_set_size_dist.png", dpi=dpi)
    return fig
    
    
def plot_qualitative_grid(samples_imgs, samples_masks, samples_probs, samples_setsize,
                            rgb_channels, class_names, palette, num_classes, viz_samples,
                            model_label, alpha, q_hat, coverage, avg_set_size,
                            viz_out_dir, model_key, dpi):
    """
    Generates a qualitative grid showing input, ground truth, prediction, set size, and uncertainty.
    """

    seg_cmap  = ListedColormap(palette[:num_classes])
    seg_norm  = BoundaryNorm(boundaries=list(range(num_classes + 1)), ncolors=num_classes)
    size_cmap = plt.cm.plasma

    def to_rgb(img_tensor, rgb_idx):
        """Extract and normalise 3 channels for display."""
        ch = img_tensor[rgb_idx]           # (3, H, W)
        lo, hi = ch.min(), ch.max()
        ch = (ch - lo) / (hi - lo + 1e-8)
        return ch.permute(1, 2, 0).numpy()

    def make_legend(ax, names, palette):
        patches = [mpatches.Patch(color=palette[i], label=names[i]) for i in range(len(names))]
        ax.legend(handles=patches, fontsize=6, loc="upper right", framealpha=0.7)

    n_show = min(viz_samples, len(samples_imgs))
    n_cols = 5   # RGB | GT | Argmax | Set-size | Uncertainty
    col_titles = ["RGB input", "Ground truth", "Argmax prediction", "Prediction set size", "Max class probability"]

    fig = plt.figure(figsize=(n_cols * 3.2, n_show * 3.0))
    gs  = gridspec.GridSpec(n_show, n_cols, figure=fig, hspace=0.35, wspace=0.08)

    for row, idx in enumerate(range(n_show)):
        img      = samples_imgs[idx]        # (ch, H, W)
        gt       = samples_masks[idx]       # (H, W)
        probs_i  = samples_probs[idx]       # (C, H, W)
        setsize  = samples_setsize[idx]     # (H, W)

        argmax   = probs_i.argmax(dim=0)    # (H, W)
        max_prob = probs_i.max(dim=0).values # (H, W)  — confidence map

        gt_disp = gt.numpy().copy().astype(float)
        gt_disp[gt_disp == 255] = np.nan   # ignore index → transparent

        # Col 0 — RGB input
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(to_rgb(img, rgb_channels))
        ax.axis("off")
        if row == 0: ax.set_title(col_titles[0], fontsize=9, fontweight="bold")

        # Col 1 — Ground truth
        ax = fig.add_subplot(gs[row, 1])
        ax.imshow(gt_disp, cmap=seg_cmap, norm=seg_norm, interpolation="nearest")
        if row == 0:
            ax.set_title(col_titles[1], fontsize=9, fontweight="bold")
            make_legend(ax, class_names, palette)
        ax.axis("off")

        # Col 2 — Argmax prediction
        ax = fig.add_subplot(gs[row, 2])
        ax.imshow(argmax.numpy(), cmap=seg_cmap, norm=seg_norm, interpolation="nearest")
        if row == 0: ax.set_title(col_titles[2], fontsize=9, fontweight="bold")
        ax.axis("off")

        # Col 3 — Set size map
        ax = fig.add_subplot(gs[row, 3])
        im = ax.imshow(setsize.numpy(), cmap=size_cmap, vmin=1, vmax=num_classes, interpolation="nearest")
        if row == 0:
            ax.set_title(col_titles[3], fontsize=9, fontweight="bold")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis("off")

        # Col 4 — Max probability (confidence)
        ax = fig.add_subplot(gs[row, 4])
        im2 = ax.imshow(max_prob.numpy(), cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
        if row == 0:
            ax.set_title(col_titles[4], fontsize=9, fontweight="bold")
            plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
        ax.axis("off")

    fig.suptitle(
        f"{model_label}  |  α={alpha}  |  q̂={q_hat:.3f}  |  "
        f"coverage={coverage:.2%}  |  avg set size={avg_set_size:.2f}",
        fontsize=10, fontweight="bold", y=1.01
    )
    fig.savefig(viz_out_dir /f"{model_key}"/ f"{model_key}_qualitative_grid.png",
                dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_multi_model_comparison(results, alpha, viz_out_dir, dpi):
    """
    Plots a comparison of marginal coverage, average prediction set size, and calibration threshold
    across multiple models.
    """
    if len(results) < 2:
        print("Need at least 2 models to compare — train more checkpoints first.")
        return

    labels   = [r["label"]        for r in results]
    coverages= [r["coverage"]     for r in results]
    set_szs  = [r["avg_set_size"] for r in results]
    q_hats   = [r["q_hat"]        for r in results]

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, vals, title, ylabel, hline in [
        (axes[0], coverages, "Marginal coverage",    "Coverage",       1 - alpha),
        (axes[1], set_szs,   "Avg prediction set size", "Set size",    1.0),
        (axes[2], q_hats,    "Calibration threshold q̂", "q̂",          None),
    ]:
        # colors = ["#2ecc71" if v >= (1 - alpha if hline == 1 - alpha else 0) else "#e74c3c" # Original logic
        #           for v in vals]
        # Simplified color logic for all bars to be 'steelblue' unless specific condition is met, but here it's for general comparison.
        # Sticking to the original color scheme, assuming it's related to meeting the target.
        colors = ["steelblue"] * len(vals) # Default color
        if hline is not None:
            # Apply color logic based on meeting the target, similar to single model plot
            colors = ["#2ecc71" if v >= hline else "#e74c3c" for v in vals]

        ax.bar(x, vals, color=colors, edgecolor="white")
        if hline is not None:
            ax.axhline(hline, color="crimson", lw=1.5, ls="--",
                       label=f"target = {hline}")
            ax.legend(fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel)

    fig.suptitle(f"Multi-model comparison  (α = {alpha})", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(viz_out_dir / "multi_model_comparison.png",
                dpi=dpi, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    confusion_matrix,
    class_names=None,
    normalize: bool = True,
    title: str | None = None,
    cmap: str = "Blues",
):
    """Plot a segmentation confusion matrix with true classes on rows."""
    matrix = np.asarray(confusion_matrix, dtype=np.float64)
    display_matrix = matrix.copy()
    if normalize:
        row_totals = display_matrix.sum(axis=1, keepdims=True)
        display_matrix = np.divide(
            display_matrix,
            row_totals,
            out=np.zeros_like(display_matrix, dtype=np.float64),
            where=row_totals != 0,
        )

    labels = [_class_name(class_names, index) for index in range(matrix.shape[0])]
    fig, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(display_matrix, interpolation="nearest", cmap=cmap)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    axis.set_title(title or "Confusion Matrix")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_xticks(np.arange(len(labels)))
    axis.set_yticks(np.arange(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_yticklabels(labels)

    threshold = display_matrix.max() / 2 if display_matrix.size else 0
    for row in range(display_matrix.shape[0]):
        for column in range(display_matrix.shape[1]):
            value = display_matrix[row, column]
            label = f"{value:.2f}" if normalize else f"{int(matrix[row, column])}"
            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8,
            )

    fig.tight_layout()
    return fig
