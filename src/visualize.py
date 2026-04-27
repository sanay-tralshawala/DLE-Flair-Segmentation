"""Visualization helpers for segmentation masks and prediction logits."""

from __future__ import annotations

import colorsys
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


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
        axis.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=min(4, len(handles)))


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
