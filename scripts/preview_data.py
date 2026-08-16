#!/usr/bin/env python
"""Render the README dataset overview grid.

Generates a 10-row (one per ImageNette class) x N-column grid of samples,
passed through the eval transform so the figure shows exactly what the model
sees.

Usage:
    python scripts/preview_data.py
    python scripts/preview_data.py --per-class 6 --seed 1 --no-labels

Note: portions of this file were generated with the assistance of
Anthropic's Claude Opus 5, then reviewed line by line by the author.
"""

import argparse
import os
import random
import sys

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import make_grid, save_image

from image_diffusion.data import (
    IMAGENETTE_CLASSES,
    _build_transforms,
    denormalize,
)

IMAGE_SIZE = 128
IMAGE_EXTS = (".jpeg", ".jpg", ".png")
PADDING = 2
LABEL_WIDTH = 190


def collect_per_class(root, split, per_class, seed):
    """Return one list of image paths per class, in ImageFolder label order.

    Returns (rows, wnids) so the caller can print the row order it got.
    """
    split_dir = os.path.join(root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"No such dir: {split_dir}")

    wnids = sorted(
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    )
    if len(wnids) != len(IMAGENETTE_CLASSES):
        raise RuntimeError(
            f"expected {len(IMAGENETTE_CLASSES)} class dirs, "
            f"found {len(wnids)} in {split_dir}"
        )

    rows = []
    for label, wnid in enumerate(wnids):
        class_dir = os.path.join(split_dir, wnid)
        files = sorted(
            f for f in os.listdir(class_dir)
            if f.lower().endswith(IMAGE_EXTS)
        )
        if len(files) < per_class:
            raise RuntimeError(
                f"{wnid} has {len(files)} images, need {per_class}"
            )
 
        random.Random(seed * 1000 + label).shuffle(files)
        rows.append([os.path.join(class_dir, f) for f in files[:per_class]])

    return rows, wnids


def load_row(paths, transform):
    """Load one class's images into a (len(paths), 3, H, W) float tensor."""
    tensors = []
    for path in paths:
        with Image.open(path) as im:
            tensors.append(transform(im.convert("RGB")))
    return torch.stack(tensors)  # n x (3, H, W) -> (n, 3, H, W)


def add_row_labels(png_path, names, tile, padding):
    """Reopen the saved grid and draw class names into a left margin."""
    grid = Image.open(png_path).convert("RGB")
    out = Image.new("RGB", (grid.width + LABEL_WIDTH, grid.height), "white")
    out.paste(grid, (LABEL_WIDTH, 0))

    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:  # Pillow < 10.1 has no size argument
        font = ImageFont.load_default()

    for row, name in enumerate(names):
        y = padding + row * (tile + padding) + tile // 2 - 9
        draw.text((12, y), name, fill="black", font=font)

    out.save(png_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/imagenette2-320")
    ap.add_argument("--split", default="train")
    ap.add_argument("--per-class", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    ap.add_argument("--out", default="docs/assets/dataset-overview.png")
    ap.add_argument("--no-labels", action="store_true")
    args = ap.parse_args()

    transform = _build_transforms("val", image_size=args.image_size)
    rows, wnids = collect_per_class(
        args.root, args.split, args.per_class, args.seed
    )

    # cat, not stack: each row is already (per_class, 3, H, W)
    batch = torch.cat([load_row(paths, transform) for paths in rows])

    expected = (len(rows) * args.per_class, 3, args.image_size, args.image_size)
    if tuple(batch.shape) != expected:
        print(f"FAIL  expected {expected}, got {tuple(batch.shape)}")
        return 1

    # nrow is images per ROW, not the number of rows.
    grid = make_grid(denormalize(batch), nrow=args.per_class, padding=PADDING)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_image(grid, args.out)

    if not args.no_labels:
        add_row_labels(args.out, IMAGENETTE_CLASSES, args.image_size, PADDING)

    print(f"{args.split}: {len(rows)} classes x {args.per_class} samples "
          f"at {args.image_size}px  (seed {args.seed})")
    for label, (wnid, name) in enumerate(zip(wnids, IMAGENETTE_CLASSES)):
        print(f"  row {label}  {wnid}  {name}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
