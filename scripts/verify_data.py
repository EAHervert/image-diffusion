"""Verify an extracted ImageNette tree. Read-only; never downloads.

Four passes:
  1. Structure: train/ and val/ exist, each holding the 10 expected classes
  2. Counts: per-class and per-split totals match known-good values
  3. Size: on-disk footprint, mean file size, zero-byte files
  4. Quality: every file opens and reports a sane resolution

Note: --deep additionally decodes and RGB-converts each image
Exit code is 0 only if every hard check passes.

Usage:
    python scripts/verify_data.py
    python scripts/verify_data.py --root data/imagenette2-320 --deep
    python scripts/verify_data.py --sample 500

Note: portions of this file were generated with the assistance of
Anthropic's Claude Opus 5, then reviewed line by line by the author.
"""

import argparse
import random
import sys
from pathlib import Path

from PIL import Image

SPLITS = ("train", "val")
IMG_EXTS = {".jpeg", ".jpg", ".png"}

# Known-good ImageNette-320 counts.
EXPECTED_TOTALS = {"train": 9469, "val": 3925}

# WNID to human-readable: sorted WNID order is exactly the label order
# ImageFolder assigns (0-9), so this table doubles as the label map.
CLASSES = {
    "n01440764": "Tench",
    "n02102040": "English springer",
    "n02979186": "Cassette player",
    "n03000684": "Chainsaw",
    "n03028079": "Church",
    "n03394916": "French horn",
    "n03417042": "Garbage truck",
    "n03425413": "Gas pump",
    "n03445777": "Golf ball",
    "n03888257": "Parachute",
}

# The pipeline resizes the shortest side to 128 and never upsamples, so
# anything below this is a data problem rather than a config choice.
MIN_SHORT_SIDE = 128


# Helper functions.
def format_bytes(n):
    """Bytes to human-readable display string."""
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    size = float(n)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024


def iter_images(split_dir):
    """Yield (wnid, path) for every image file under a split directory."""
    if not split_dir.is_dir():
        return
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() in IMG_EXTS:
                yield class_dir.name, f


# Test 1: structure
def check_structure(root):
    print("\nTest [1/4]: structure")
    if not root.is_dir():
        print(f"  FAIL: no such directory: {root}")
        return False

    ok = True
    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            print(f"  FAIL: missing split directory: {split_dir}")
            ok = False
            continue

        found = set([p.name for p in split_dir.iterdir() if p.is_dir()])
        missing = sorted(set(CLASSES) - found)
        extra = sorted(found - set(CLASSES))

        if missing:
            print(f"  FAIL: {split}/ missing {len(missing)} class dirs: {missing}")
            ok = False
        if extra:
            print(f"  WARN: {split}/ has unexpected dirs: {extra}")
        if not missing and not extra:
            print(f"  ok: {split}/ has all 10 expected class dirs")
    return ok


# Test 2: counts
def check_counts(root):
    print("\nTest [2/4]: counts")
    ok = True

    for split in SPLITS:
        per_class = {}
        for wnid, _ in iter_images(root / split):
            if wnid not in per_class:
                per_class[wnid] = 0
            per_class[wnid] += 1

        # Formatting count information
        print(f"  {'split':<6} {'wnid':<11} {'class':<18} {'images':>7}")
        print("  " + "-" * 45)
        for wnid in sorted(CLASSES):
            print(f"  {split:<6} {wnid:<11} {CLASSES[wnid]:<18} "
                  f"{per_class.get(wnid, 0):>7}")

        total = sum(per_class.values())
        expected = EXPECTED_TOTALS[split]
        status = (f"ok ({total})" if total == expected
                  else f"MISMATCH (got {total}, expected {expected})")
        print("  " + "-" * 45)
        print(f"  {split}: {status}")
        ok = ok and total == expected

    return ok


# Test 3: size
def check_size(root):
    print("\nTest [3/4]: size")
    total_bytes, n, empty = 0, 0, []

    for split in SPLITS:
        for _, path in iter_images(root / split):
            size = path.stat().st_size
            total_bytes += size
            n += 1
            if size == 0:
                empty.append(path)

    if n == 0:
        print("  FAIL: no image files found")
        return False

    print(f"  files     : {n}")
    print(f"  on disk   : {format_bytes(total_bytes)}")
    print(f"  mean file : {format_bytes(total_bytes / n)}")

    if empty:
        print(f"  FAIL: {len(empty)} zero-byte files, e.g. {empty[0]}")
        return False

    print("  ok: no zero-byte files")
    return True


# Test 4: quality
def check_quality(root, deep=False, sample=None, seed=0):
    label = "  (deep: full decode)" if deep else ""
    print(f"\nTest [4/4]: quality{label}")

    paths = [p for split in SPLITS for _, p in iter_images(root / split)]
    if sample and sample < len(paths):
        random.Random(seed).shuffle(paths)
        paths = paths[:sample]
        print(f"  sampling {len(paths)} images")

    corrupt, undersized, non_rgb, short_sides = [], [], [], []

    for path in paths:
        try:
            with Image.open(path) as im:
                w, h = im.size  # header only; no pixel decode yet
                mode = im.mode
                if deep:
                    im.convert("RGB").load()  # full decode
                else:
                    im.verify()  # cheap header/structure check
        except Exception as exc:
            corrupt.append((path, repr(exc)))
            continue

        short_sides.append(min(w, h))
        if min(w, h) < MIN_SHORT_SIDE:
            undersized.append((path, w, h))
        if mode != "RGB":
            non_rgb.append((path, mode))

    if short_sides:
        short_sides.sort()
        p50 = short_sides[len(short_sides) // 2]
        print(f"  short side min/p50    : {short_sides[0]} / {p50}")

    # Grayscale and CMYK source images are normal in ImageNet-derived data;
    # ImageFolder converts them at load time. Informational, not a failure.
    print(f"  non-RGB source modes  : {len(non_rgb)}")

    ok = True
    if corrupt:
        print(f"  FAIL: {len(corrupt)} unreadable files")
        for path, err in corrupt[:5]:
            print(f"        {path}  {err}")
        ok = False
    else:
        print("  ok: every checked file opens cleanly")

    if undersized:
        print(f"  FAIL: {len(undersized)} images below {MIN_SHORT_SIDE}px short side")
        for path, w, h in undersized[:5]:
            print(f"        {path}  {w}x{h}")
        ok = False
    else:
        print(f"  ok: nothing below {MIN_SHORT_SIDE}px on the short side")

    return ok


# Entry point
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/imagenette2-320")
    ap.add_argument("--deep", action="store_true",
                    help="fully decode every image (slow; catches truncation)")
    ap.add_argument("--sample", type=int, default=None,
                    help="check only N random images in the quality pass")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)

    results = {"structure": check_structure(root)}
    if not results["structure"]:
        print("\nstructure check failed - stopping early")
        return 1

    results["counts"] = check_counts(root)
    results["size"] = check_size(root)
    results["quality"] = check_quality(root, args.deep, args.sample, args.seed)

    print("\n" + "=" * 30)
    for name, passed in results.items():
        print(f"  {name:<10} {'PASS' if passed else 'FAIL'}")
    print("=" * 30)

    if all(results.values()):
        print(f"\ndataset verified at {root}")
        return 0

    print(f"\nverification FAILED for {root}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
