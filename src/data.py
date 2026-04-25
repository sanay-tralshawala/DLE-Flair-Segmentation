"""Dataset, dataloader, mask remapping, and augmentation helpers."""

import torch


FLAIR_5CLASS_NAMES = {
    0: "trees/bush",
    1: "vegetation",
    2: "soil/land",
    3: "building/manmade",
    4: "other",
}

FLAIR_19_TO_5CLASS = {
    1: 3,   # building -> building/manmade
    2: 2,   # pervious surface -> soil/land
    3: 3,   # impervious surface -> building/manmade
    4: 2,   # bare soil -> soil/land
    5: 4,   # water -> other
    6: 0,   # coniferous -> trees/bush
    7: 0,   # deciduous -> trees/bush
    8: 0,   # brushwood -> trees/bush
    9: 1,   # vineyard -> vegetation
    10: 1,  # herbaceous vegetation -> vegetation
    11: 1,  # agricultural land -> vegetation
    12: 2,  # plowed land -> soil/land
    13: 3,  # swimming pool -> building/manmade
    14: 4,  # snow -> other
    15: 4,  # clear cut -> other
    16: 0,  # mixed -> trees/bush
    17: 0,  # ligneous -> trees/bush
    18: 3,  # greenhouse -> building/manmade
    19: 4,  # other -> other
}


def load_image_tensor(image_path, input_channels: str = "grayscale") -> torch.Tensor:
    """Return an image tensor shaped [C, H, W], float32, ready for augmentation."""
    raise NotImplementedError


def load_mask_tensor(mask_path) -> torch.Tensor:
    """Return the original FLAIR mask tensor shaped [H, W] with integer class IDs."""
    raise NotImplementedError


def remap_mask_to_5class(mask: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    """Map original FLAIR IDs to contiguous 5-class target IDs in [0, 4]."""
    remapped = torch.full_like(mask.long(), fill_value=ignore_index)
    for source_class, target_class in FLAIR_19_TO_5CLASS.items():
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


def build_dataloaders(data_config: dict):
    """Return train, val, and test dataloaders yielding dicts with `image` and `mask`."""
    raise NotImplementedError
