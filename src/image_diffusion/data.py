"""
Dataset builders for image-diffusion.

Datasets in scope:
- ImageNette-128 - 10-class ImageNet subset. ~13k images.

Augmentation pipeline:
- Train: RandomResizedCrop(image_size), RandomHorizontalFlip, ToTensor, Normalize to [-1, 1]
- Val:   Resize(int(image_size * 1.14)), CenterCrop(image_size), ToTensor, Normalize to [-1, 1]

Normalization to [-1, 1] matches the noise ~ N(0, I) used in flow matching.
"""

import os

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


# Normalize [0, 1] -> [-1, 1] via (x - 0.5) / 0.5, applied per channel.
IMAGENETTE_MEAN = (0.5, 0.5, 0.5)
IMAGENETTE_STD  = (0.5, 0.5, 0.5)

# ImageNette WNID -> human-readable class name, in the alphabetical order
# that ImageFolder uses to assign integer labels 0..9.
IMAGENETTE_CLASSES = (
    "Tench",             # n01440764
    "English springer",  # n02102040
    "Cassette player",   # n02979186
    "Chainsaw",          # n03000684
    "Church",            # n03028079
    "French horn",       # n03394916
    "Garbage truck",     # n03417042
    "Gas pump",          # n03425413
    "Golf ball",         # n03445777
    "Parachute",         # n03888257
)


def _build_transforms(split, image_size=128) -> transforms.Compose:
    """
    Helper function for the augmentation/normalization pipeline.
    Returns torchvision.Compose object.
    """

    if split == 'train':
        return transforms.Compose([transforms.RandomResizedCrop(image_size),
                                   transforms.RandomHorizontalFlip(), transforms.ToTensor(), 
                                   transforms.Normalize(mean=IMAGENETTE_MEAN, std=IMAGENETTE_STD)])
    elif split == 'val':
        return transforms.Compose([transforms.Resize(int(image_size * 1.14)),
                                   transforms.CenterCrop(image_size), transforms.ToTensor(),
                                   transforms.Normalize(mean=IMAGENETTE_MEAN, std=IMAGENETTE_STD)])
    else:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")


def build_imagenette_loader(
    root: str,
    split: str,
    batch_size: int,
    num_workers: int = 4,
    image_size: int = 128,
    shuffle: bool | None = None,
    drop_last: bool | None = None,
) -> DataLoader:
    """
    Build a DataLoader for ImageNette-{split}.

    Args:
        root: path to the imagenette2 directory (train or val split).
        split: 'train' or 'val'.
        batch_size: samples per batch.
        num_workers: parallel data-loading workers.
        image_size: crop size in pixels.
        shuffle: overrides split-based default (train=True, val=False).
        drop_last: overrides split-based default (train=True, val=False).

    Returns:
        A DataLoader yielding (images, labels) tuples with
        images shape (B, 3, image_size, image_size), float32, values in [-1, 1],
        labels shape (B,), long, values in [0, 9].
    """

    data_dir = os.path.join(root, split)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"ImageNette split directory not found: {os.path.abspath(data_dir)!r}. "
            f"Expected root/{split}/ with 10 class subdirectories."
        )

    shuffle   = (split == 'train') if shuffle is None else shuffle
    drop_last = (split == 'train') if drop_last is None else drop_last

    image_transforms = _build_transforms(split=split, image_size=image_size)
    dataset = ImageFolder(root=data_dir, transform=image_transforms)
    if len(dataset.classes) != 10:
        raise RuntimeError(
            f"Expected 10 ImageNette classes at {data_dir!r}, "
            f"found {len(dataset.classes)}: {dataset.classes}"
        )

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                    num_workers=num_workers, persistent_workers=(num_workers > 0))
