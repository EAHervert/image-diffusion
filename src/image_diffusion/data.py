"""
Dataset builders for image-diffusion.

Datasets in scope:
- ImageNette-128 - 10-class ImageNet subset. ~13k images.

Augmentation pipeline:
- Train:     Resize(image_size), RandomCrop(image_size), RandomHorizontalFlip
- Val/Test:  Resize(image_size), CenterCrop(image_size)
- Both then: ToTensor, Normalize -> [-1, 1]

Note: ImageNette ships only train/ and val/. The 'val' and 'test' splits are
carved deterministically out of the canonical val/ directory by a
stratified per-class index selection.

Normalization to [-1, 1] matches the noise ~ N(0, I) used in flow matching.
"""

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

SPLITS = ("train", "val", "test")

# Normalize [0, 1] -> [-1, 1] via (x - 0.5) / 0.5, applied per channel.
IMAGENETTE_SHIFT = (0.5, 0.5, 0.5)
IMAGENETTE_SCALE = (0.5, 0.5, 0.5)

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


def _stratified_split_indices(targets, frac, seed):
    """Deterministic per-class split of dataset indices.

    Returns (indices_a, indices_b) where indices_a holds `frac` of each
    class, chosen by a fixed-seed permutation of the sorted index list.
    Depends only on (targets, frac, seed).
    """
    by_class = {}
    for idx, target in enumerate(targets):
        by_class.setdefault(int(target), []).append(idx)

    a, b = [], []
    rng = random.Random(seed)
    for target in sorted(by_class):
        idxs = sorted(by_class[target])
        rng.shuffle(idxs)
        k = int(frac * len(idxs))
        a.extend(idxs[:k])
        b.extend(idxs[k:])

    return sorted(a), sorted(b)


def _seed_worker(_worker_id):
    """Give each DataLoader worker a deterministic, distinct RNG state."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def denormalize(x):
    """Map a normalized tensor back to [0, 1] for viewing or saving.

    Inverse of the affine map _build_transforms applies. Derived from
    IMAGENETTE_SHIFT/SCALE rather than hardcoding 0.5 so the forward and
    inverse maps cannot drift apart. Accepts (C, H, W) or (B, C, H, W).
    """
    shift = torch.tensor(IMAGENETTE_SHIFT, dtype=x.dtype, device=x.device)
    scale = torch.tensor(IMAGENETTE_SCALE, dtype=x.dtype, device=x.device)
    return (x * scale.view(-1, 1, 1) + shift.view(-1, 1, 1)).clamp(0, 1)


def _build_transforms(split, image_size=128, hflip=True) -> transforms.Compose:
    """Augmentation/normalization pipeline for a split."""
    if split == "train":
        head = [
            transforms.Resize(image_size),      # int => shortest side, aspect kept
            transforms.RandomCrop(image_size),  # slides along the long axis
        ]
        if hflip:
            head.append(transforms.RandomHorizontalFlip())
    elif split in ("val", "test"):
        head = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
    else:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    return transforms.Compose(head + [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENETTE_SHIFT, std=IMAGENETTE_SCALE),
    ])


def build_imagenette_loader(
    root: str,
    split: str,
    batch_size: int,
    num_workers: int = 4,
    image_size: int = 128,
    hflip: bool = True,
    shuffle: bool | None = None,
    drop_last: bool | None = None,
    seed: int = 0,
    split_seed: int = 1234,
    val_test_frac: float = 0.5,
    pin_memory: bool | None = None,
) -> DataLoader:
    """
    Build a DataLoader for ImageNette-{split}.

    Args:
        root: path to the imagenette2-320 directory.
        split: 'train', 'val', or 'test'.
        batch_size: samples per batch.
        num_workers: parallel data-loading workers.
        image_size: crop size in pixels.
        hflip: apply RandomHorizontalFlip.
        shuffle: overrides split-based default (train=True, else False).
        drop_last: overrides split-based default (train=True, else False).
        seed: seeds the DataLoader shuffle generator and the workers.
        split_seed: seeds the val/test carve. Changing it reshuffles the split.
        val_test_frac: fraction of val/ assigned to 'val'; remainder to 'test'.
        pin_memory: defaults to True only on CUDA (a no-op that warns on MPS).

    Returns:
        A DataLoader yielding (images, labels) tuples with
        images shape (B, 3, image_size, image_size), float32, values in [-1, 1],
        labels shape (B,), long, values in [0, 9].
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    # 'test' has no directory of its own - it is carved out of val/.
    disk_split = "train" if split == "train" else "val"
    data_dir = os.path.join(root, disk_split)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"ImageNette split directory not found: {os.path.abspath(data_dir)!r}. "
            f"Expected root/{disk_split}/ with 10 class subdirectories."
        )

    shuffle   = (split == "train") if shuffle is None else shuffle
    drop_last = (split == "train") if drop_last is None else drop_last
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    image_transforms = _build_transforms(split, image_size=image_size, hflip=hflip)
    dataset = ImageFolder(root=data_dir, transform=image_transforms)
    if len(dataset.classes) != 10:
        raise RuntimeError(
            f"Expected 10 ImageNette classes at {data_dir!r}, "
            f"found {len(dataset.classes)}: {dataset.classes}"
        )

    # Carve val/ into val + test. Done AFTER the class check, since a
    # Subset exposes no .classes attribute.
    if split in ("val", "test"):
        val_idx, test_idx = _stratified_split_indices(
            dataset.targets, val_test_frac, split_seed
        )
        dataset = Subset(dataset, val_idx if split == "val" else test_idx)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=pin_memory,
        generator=generator,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
    )
