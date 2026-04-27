"""Dataset, dataloader, mask remapping, and augmentation helpers."""

import csv
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset


def load_class_map(class_map_path) -> dict:
    """Load class names, ignore index, and original-to-target mask mapping from 01_eda notebook."""
    with open(class_map_path) as file:
        class_map = json.load(file)
    class_map["source_to_target"] = {
        int(source_id): int(target_id)
        for source_id, target_id in class_map["source_to_target"].items()
    }
    return class_map


def _resolve_path(path, repo_root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root / path


def _repo_root() -> Path:
    # simple helper to get root path
    return Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent


def normalize_image(
    image: np.ndarray,
    norm_type: str = "scaling",
    means: list[float] | None = None,
    stds: list[float] | None = None,
) -> np.ndarray:
    """
    Normalize a [C, H, W] image using FLAIR-style `scaling`, `custom`, or `without`.
    Logic pulled directly from FLAIR base code's `norm_type` options, with `custom` means/stds passed in from config.
    """
    if norm_type == "without":
        return image.astype(np.float32)

    if norm_type == "scaling":
        # Mirrors FLAIR base code's `norm_type: scaling`, without adding skimage as a dependency.
        if np.issubdtype(image.dtype, np.integer):
            return image.astype(np.float32) / np.iinfo(image.dtype).max
        return image.astype(np.float32)

    if norm_type == "custom":
        means = means or []
        stds = stds or []
        image = image.astype(np.float32)
        for channel_index, (mean, std) in enumerate(zip(means, stds)):
            image[channel_index] = (image[channel_index] - mean) / std
        return image

    raise ValueError(f"Unknown norm_type: {norm_type}")


def load_image_tensor(
    image_path,
    channels: list[int],
    norm_type: str = "scaling",
    means: list[float] | None = None,
    stds: list[float] | None = None,
) -> torch.Tensor:
    """Read FLAIR image channels with rasterio and return [C, H, W] float32 tensor."""
    # Using FLAIR base code's rasterio read pattern; channels are 1-indexed.
    with rasterio.open(image_path) as src_img:
        image = src_img.read(channels)
    image = normalize_image(image, norm_type=norm_type, means=means, stds=stds)
    return torch.as_tensor(image, dtype=torch.float32)


def load_mask_tensor(mask_path) -> torch.Tensor:
    """Read original FLAIR mask with rasterio and return [H, W] integer tensor."""
    # Using FLAIR base code's mask read pattern: first raster band contains class IDs.
    with rasterio.open(mask_path) as src_msk:
        mask = src_msk.read()[0]
    return torch.as_tensor(mask, dtype=torch.long)


def remap_mask(mask: torch.Tensor, class_map: dict) -> torch.Tensor:
    """Map original FLAIR IDs to contiguous target IDs using class_map.json."""
    ignore_index = class_map.get("ignore_index", 255)
    remapped = torch.full_like(mask.long(), fill_value=ignore_index)
    for source_class, target_class in class_map["source_to_target"].items():
        remapped[mask == source_class] = target_class
    return remapped


def build_train_augmentation(augment_config: dict):
    """Return a callable that applies shared v2 flips and rot90 to image/mask tensors."""
    from torchvision.transforms import v2

    if not augment_config.get("enabled", False):
        return lambda image, mask: (image, mask)

    transforms = []
    if augment_config.get("horizontal_flip", False):
        transforms.append(v2.RandomHorizontalFlip(p=0.5))
    if augment_config.get("vertical_flip", False):
        transforms.append(v2.RandomVerticalFlip(p=0.5))

    flip_transform = v2.Compose(transforms) if transforms else None
    use_rot90 = augment_config.get("random_rotate_90", False)

    def augment(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if flip_transform is not None:
            image, mask = flip_transform(image, mask)
        if use_rot90:
            k = int(torch.randint(0, 4, ()).item())
            image = torch.rot90(image, k, dims=(-2, -1))
            mask = torch.rot90(mask, k, dims=(-2, -1))
        return image, mask

    return augment


def load_split_records(csv_path, repo_root: Path | None = None) -> list[dict]:
    """Read a split CSV with `image_path` and `mask_path` columns into absolute paths."""
    repo_root = repo_root or _repo_root()
    records = []
    with open(csv_path, newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            records.append({
                "image_path": _resolve_path(row["image_path"], repo_root),
                "mask_path": _resolve_path(row["mask_path"], repo_root),
                "original_split": row.get("original_split"),
            })
    return records


class FlairSegmentationDataset(Dataset):
    """FLAIR patch dataset returning train.py-compatible `image` and `mask` tensors."""

    def __init__(
        self,
        records: list[dict],
        class_map: dict,
        data_config: dict,
        augment=None,
    ):
        self.records = records
        self.class_map = class_map
        self.channels = data_config["channels"]
        self.norm_type = data_config.get("norm_type", "scaling")
        self.norm_means = data_config.get("norm_means", [])
        self.norm_stds = data_config.get("norm_stds", [])
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image = load_image_tensor(
            record["image_path"],
            channels=self.channels,
            norm_type=self.norm_type,
            means=self.norm_means,
            stds=self.norm_stds,
        )
        mask = load_mask_tensor(record["mask_path"])
        mask = remap_mask(mask, self.class_map)

        if self.augment is not None:
            image, mask = self.augment(image, mask)

        return {"image": image, "mask": mask}

# quick dataloader to test dataset and augmentation logic; plan to use sanays notebook version at first
def build_dataloaders(data_config: dict):
    """Return train, val, and test dataloaders yielding dicts with `image` and `mask`."""
    repo_root = _repo_root()
    class_map = load_class_map(_resolve_path(data_config["class_map"], repo_root))

    train_records = load_split_records(_resolve_path(data_config["train_csv"], repo_root), repo_root)
    val_records = load_split_records(_resolve_path(data_config["val_csv"], repo_root), repo_root)
    test_records = load_split_records(_resolve_path(data_config["test_csv"], repo_root), repo_root)

    train_dataset = FlairSegmentationDataset(
        train_records,
        class_map=class_map,
        data_config=data_config,
        augment=build_train_augmentation(data_config["augment"]),
    )
    val_dataset = FlairSegmentationDataset(val_records, class_map=class_map, data_config=data_config)
    test_dataset = FlairSegmentationDataset(test_records, class_map=class_map, data_config=data_config)

    loader_kwargs = {
        "batch_size": data_config["batch_size"],
        "num_workers": data_config["num_workers"],
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader
