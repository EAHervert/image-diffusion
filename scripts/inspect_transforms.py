"""Inspect the ImageNette resize/crop pipeline.

Run this BEFORE hardening data.py. It measures what the augmentation
actually does to real images, rather than trusting the config.

Usage:
    python scripts/inspect_transforms.py --root data/imagenette2-320
    python scripts/inspect_transforms.py --root data/imagenette2-320 --n 500

Note: portions of this file were generated with the assistance of
Anthropic's Claude Opus 5, then reviewed line by line by the author.
"""

import argparse
import os
import random

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid, save_image

IMAGE_SIZE = 128

# These are the shift and scale of an affine map:
#   out = (in - SHIFT) / SCALE = 2 * in - 1, [0, 1] -> [-1, 1].
SHIFT = (0.5, 0.5, 0.5)
SCALE = (0.5, 0.5, 0.5)

# RandomResizedCrop configs, kept ONLY as diagnostics.
RRC_CANDIDATES = {
    "RRC default    scale=(0.08,1.0) ratio=(3/4,4/3)": dict(scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3)),
    "RRC tightened  scale=(0.80,1.0) ratio=(0.9,1.11)": dict(scale=(0.80, 1.0), ratio=(0.90, 1.11)),
}


def collect_paths(root, split="train", limit=None, seed=0):
    """Gather image paths from root/split/<class>/*.JPEG, deterministically."""
    split_dir = os.path.join(root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"No such dir: {os.path.abspath(split_dir)}")

    paths = []
    for class_dir in sorted(os.listdir(split_dir)):
        full = os.path.join(split_dir, class_dir)
        if not os.path.isdir(full):
            continue
        for fname in sorted(os.listdir(full)):
            if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                paths.append(os.path.join(full, fname))

    rng = random.Random(seed)
    rng.shuffle(paths)
    return paths[:limit] if limit else paths


def pct(values, q):
    """Percentile of a python list, no numpy needed."""
    t = torch.tensor(values, dtype=torch.float32)
    return torch.quantile(t, q).item()


def report_raw_resolution(paths):
    """Q1: what are we actually starting from?"""
    widths, heights, short_sides, aspects = [], [], [], []
    for p in paths:
        with Image.open(p) as im:
            w, h = im.size
        widths.append(w)
        heights.append(h)
        short_sides.append(min(w, h))
        aspects.append(max(w, h) / min(w, h))

    print("\nQ1. Raw resolution")
    print(f"  images sampled     : {len(paths)}")
    print(f"  shortest side  min : {min(short_sides)}")
    print(f"  shortest side  p50 : {pct(short_sides, 0.50):.0f}")
    print(f"  shortest side  max : {max(short_sides)}")
    print(f"  aspect ratio   p50 : {pct(aspects, 0.50):.2f}")
    print(f"  aspect ratio   p99 : {pct(aspects, 0.99):.2f}")

    below = sum(s < IMAGE_SIZE for s in short_sides)
    print(f"  shorter than {IMAGE_SIZE}px : {below} ({100 * below / len(paths):.1f}%)")
    return short_sides


def report_crop_behavior(paths, name, scale, ratio, samples_per_image=4, seed=0):
    """Q2: upsampling. Q3: distortion + frame coverage. Q4: fallback rate."""
    torch.manual_seed(seed)
    area_fracs, crop_short_sides, distortions = [], [], []
    upsamples, fallback_images = 0, 0

    for p in paths:
        with Image.open(p) as im:
            im = im.convert("RGB")
            W, H = im.size  # PIL is (width, height)
            orig_area = W * H
            boxes = []
            for _ in range(samples_per_image):
                i, j, h, w = transforms.RandomResizedCrop.get_params(im, scale, ratio)
                boxes.append((i, j, h, w))
                area_fracs.append((h * w) / orig_area)
                crop_short_sides.append(min(h, w))
                # The box is squashed into a SQUARE output,
                # so any non-1 box aspect is pure distortion.
                distortions.append(max(w / h, h / w))
                if min(h, w) < IMAGE_SIZE:
                    upsamples += 1
            # Identical boxes across every attempt => get_params exhausted its
            # 10 tries and returned the deterministic center-crop fallback.
            if len(set(boxes)) == 1:
                fallback_images += 1

    n = len(area_fracs)
    print(f"\n{name}")
    print(f"  crops sampled         : {n}")
    print(f"  area kept        p05  : {pct(area_fracs, 0.05):.1%}")
    print(f"  area kept        p50  : {pct(area_fracs, 0.50):.1%}")
    print(f"  area kept        p95  : {pct(area_fracs, 0.95):.1%}")
    print(f"  crop short side  p05  : {pct(crop_short_sides, 0.05):.0f}px")
    print(f"  aspect distortion p95 : {pct(distortions, 0.95):.2f}x   <-- want 1.00")
    print(f"  UPSAMPLED crops       : {upsamples} ({100 * upsamples / n:.2f}%)   <-- want 0%")
    # Only flag the fallback when it actually fired.
    note = "   <-- augmentation is DEAD here" if fallback_images else ""
    print(f"  FALLBACK images       : {fallback_images}/{len(paths)} "
          f"({100 * fallback_images / len(paths):.1f}%)" + note)


def _tail():
    """Shared terminal stages: PIL -> float tensor [0,1] -> affine map to [-1,1]."""
    return [
        transforms.ToTensor(),
        transforms.Normalize(mean=SHIFT, std=SCALE),
    ]


def make_transform(mode="train", scale=None, ratio=None):
    """Build a transform chain.

    "train"  Resize(short side) -> RandomCrop -> hflip
    "eval"   Resize(short side) -> CenterCrop
    "rrc"    RandomResizedCrop -> hflip

    Resize(int) scales the SHORTEST side to IMAGE_SIZE preserving aspect,
    then the crop slides a square window along the long axis. No aspect
    distortion, no upsampling, no silent fallback path.
    
    "rrc" exists only so inspect_transforms can measure what RandomResizedCrop
    does to real images.
    """

    if mode in ("train", "eval") and (scale is not None or ratio is not None):
        raise ValueError(f"scale/ratio are meaningless for mode={mode!r}")

    if mode == "train":
        head = [
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip()
        ]

    elif mode == "eval":
        head = [
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE)
        ]

    elif mode == "rrc":
        if scale is None or ratio is None:
            raise ValueError("mode='rrc' requires both scale and ratio")
        head = [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip()
        ]

    else:
        raise ValueError(f"unknown mode: {mode!r}")

    return transforms.Compose(head + _tail())


def denorm(x):
    """Inverse of the SHIFT/SCALE map: [-1, 1] -> [0, 1] for viewing."""
    return (x * 0.5 + 0.5).clamp(0, 1)


def report_tensor_stats(paths, seed=0, batch_size=256):
    """Sanity-check the numeric range AND predict step-0 training loss."""
    torch.manual_seed(seed)
    train_transform = make_transform()
    batch = torch.stack([train_transform(Image.open(p).convert("RGB")) for p in paths[:batch_size]])

    print("\nTensor stats (train transform)")
    print(f"  shape  : {tuple(batch.shape)}")
    print(f"  dtype  : {batch.dtype}")
    print(f"  min    : {batch.min():.3f}   (want >= -1.0)")
    print(f"  max    : {batch.max():.3f}   (want <=  1.0)")
    print(f"  mean   : {batch.mean():.3f}")
    print(f"  std    : {batch.std():.3f}")
    print(f"  finite : {bool(torch.isfinite(batch).all())}")

    # Flow matching: at init the zero-init head predicts v_hat = 0
    e_x1_sq = (batch ** 2).mean().item()
    print(f"\n  E[x_1^2]            : {e_x1_sq:.3f}")
    print(f"  PREDICTED step-0 loss: {e_x1_sq + 1.0:.3f}")


def save_comparison(paths, out, n_images=6, seed=0):
    """One row per source image: val | aggressive x2 | train x2."""
    torch.manual_seed(seed)
    val_transform = make_transform(mode="eval")
    default_transform = make_transform("rrc", **RRC_CANDIDATES["RRC default    scale=(0.08,1.0) ratio=(3/4,4/3)"])
    train_transform = make_transform()

    rows = []
    for p in paths[:n_images]:
        im = Image.open(p).convert("RGB")
        rows += [val_transform(im), default_transform(im), default_transform(im), train_transform(im), train_transform(im)]

    grid = make_grid(denorm(torch.stack(rows)), nrow=5, padding=2)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    save_image(grid, out)
    print(f"\n  wrote {out}")
    print("  columns: val(center) | aggressive | aggressive | train | train")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/imagenette2-320")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=256, help="images to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="docs/assets/transform-check.png")
    args = ap.parse_args()

    paths = collect_paths(args.root, args.split, limit=args.n, seed=args.seed)

    report_raw_resolution(paths)
    print("\nQ2/Q3/Q4. Crop behavior")
    for name, cfg in RRC_CANDIDATES.items():
        report_crop_behavior(paths, name, seed=args.seed, **cfg)
    report_tensor_stats(paths, seed=args.seed)
    save_comparison(paths, args.out, seed=args.seed)


if __name__ == "__main__":
    main()
