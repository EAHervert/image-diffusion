#!/usr/bin/env python
"""Verify the ImageNette train/val/test carve.

Five passes:
  1. Split sizes match the expected 9469 / 1959 / 1966.
  2. val and test are disjoint and exhaust the canonical val/ directory.
  3. Every class splits within one image of val_test_frac.
  4. The carve is identical across repeated calls, and moves when the seed does.
  5. One batch per split has the expected shape, dtype, and value range.

Exits 1 if any pass fails.

Usage:
    python scripts/verify_splits.py
    python scripts/verify_splits.py --root data/imagenette2-320 --image-size 128

Note: portions of this file were generated with the assistance of
Anthropic's Claude Opus 5, then reviewed line by line by the author.
"""

import argparse
import os
import sys
from collections import Counter

import torch
from torchvision.datasets import ImageFolder

# Diagnostic script: reaches into a private helper on purpose, so the carve
# can be checked directly without going through a DataLoader.
from image_diffusion.data import (
    IMAGENETTE_CLASSES,
    _stratified_split_indices,
    build_imagenette_loader,
)

EXPECTED_SIZES = {"train": 9469, "val": 1959, "test": 1966}


def _fmt(ok):
    return "PASS" if ok else "FAIL"


def _val_targets(root):
    """Labels of the canonical val/ directory, no transform, no loading."""
    return ImageFolder(os.path.join(root, "val")).targets


def pass_1_sizes(root, image_size):
    print("\n[1/5] Split sizes")
    ok = True
    for split in ("train", "val", "test"):
        loader = build_imagenette_loader(
            root, split, batch_size=8, num_workers=0, image_size=image_size
        )
        n = len(loader.dataset)
        want = EXPECTED_SIZES[split]
        hit = n == want
        ok = ok and hit  # Will be false if one of the hits is false
        print(f"  {split:>5}: {n:>5} samples   expected {want:>5}   {_fmt(hit)}")
    return ok


def pass_2_partition(root, frac, seed):
    print("\n[2/5] Partition of val/")
    targets = _val_targets(root)
    val_idx, test_idx = _stratified_split_indices(targets, frac, seed)

    disjoint = set(val_idx).isdisjoint(test_idx)
    exhaustive = sorted(val_idx + test_idx) == list(range(len(targets)))

    print(f"  val n={len(val_idx)}  test n={len(test_idx)}  source n={len(targets)}")
    print(f"  disjoint (no image in both)      {_fmt(disjoint)}")
    print(f"  exhaustive (no image dropped)    {_fmt(exhaustive)}")
    return disjoint and exhaustive


def pass_3_stratification(root, frac, seed):
    print("\n[3/5] Per-class balance")
    targets = _val_targets(root)
    val_idx, test_idx = _stratified_split_indices(targets, frac, seed)
    val_counts = Counter(targets[i] for i in val_idx)
    test_counts = Counter(targets[i] for i in test_idx)

    ok = True
    print(f"  {'class':<18}{'val':>6}{'test':>6}{'total':>7}")
    for label, name in enumerate(IMAGENETTE_CLASSES):
        v, t = val_counts[label], test_counts[label]
        want = int(frac * (v + t))
        hit = v == want
        ok = ok and hit
        print(f"  {name:<18}{v:>6}{t:>6}{v + t:>7}   {_fmt(hit)}")

    print(f"  {'TOTAL':<18}{len(val_idx):>6}{len(test_idx):>6}{len(targets):>7}")
    return ok


def pass_4_determinism(root, frac, seed):
    print("\n[4/5] Determinism")
    targets = _val_targets(root)

    a1, b1 = _stratified_split_indices(targets, frac, seed)
    a2, b2 = _stratified_split_indices(targets, frac, seed)
    stable = a1 == a2 and b1 == b2

    a3, _ = _stratified_split_indices(targets, frac, seed + 1)
    moves = a3 != a1

    print(f"  same seed -> same carve          {_fmt(stable)}")
    print(f"  seed + 1  -> different carve     {_fmt(moves)}")
    return stable and moves


def pass_5_batch(root, image_size, batch_size):
    print("\n[5/5] One batch per split")
    ok = True
    for split in ("train", "val", "test"):
        loader = build_imagenette_loader(
            root, split, batch_size=batch_size, num_workers=0, image_size=image_size
        )
        x, y = next(iter(loader))
        checks = {
            "shape": tuple(x.shape) == (batch_size, 3, image_size, image_size),
            "float32": x.dtype == torch.float32,
            "in [-1,1]": x.min() >= -1.0 and x.max() <= 1.0,
            "labels 0-9": int(y.min()) >= 0 and int(y.max()) <= 9,
        }
        ok = ok and all(checks.values())
        detail = "  ".join(f"{k} {_fmt(v)}" for k, v in checks.items())
        print(f"  {split:>5}: {tuple(x.shape)}  "
              f"min {x.min():+.3f}  max {x.max():+.3f}")
        print(f"         {detail}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/imagenette2-320")
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--split-seed", type=int, default=1234)
    ap.add_argument("--val-test-frac", type=float, default=0.5)
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"No such dir: {os.path.abspath(args.root)!r}")

    print(f"root       {os.path.abspath(args.root)}")
    print(f"image_size {args.image_size}   split_seed {args.split_seed}   "
          f"val_test_frac {args.val_test_frac}")

    results = [
        ("sizes",          pass_1_sizes(args.root, args.image_size)),
        ("partition",      pass_2_partition(args.root, args.val_test_frac,
                                            args.split_seed)),
        ("stratification", pass_3_stratification(args.root, args.val_test_frac,
                                                 args.split_seed)),
        ("determinism",    pass_4_determinism(args.root, args.val_test_frac,
                                              args.split_seed)),
        ("batch",          pass_5_batch(args.root, args.image_size,
                                        args.batch_size)),
    ]

    print("\n" + "-" * 46)
    for name, ok in results:
        print(f"  {name:<16} {_fmt(ok)}")

    failed = [name for name, ok in results if not ok]
    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
        sys.exit(1)
    print("\nAll five passes PASS.")


if __name__ == "__main__":
    main()