"""Download the ImageNette-320 tarball: extract and integrity-check.
- Idempotent: safe to re-run

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --root data --force

Note: portions of this file were generated with the assistance of
Anthropic's Claude Opus 5, then reviewed line by line by the author.
"""

import argparse
import hashlib
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

# URL Path and naming
URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
ARCHIVE_NAME = "imagenette2-320.tgz"
EXTRACT_NAME = "imagenette2-320"

EXPECTED_SHA256 = None  # TBD - e.g. "3df6f0d01f2c9e0c1b8f..."
CHUNK = 2 ** 20  # 1 MiB


# Helper functions:
def format_bytes(n):
    """Bytes to human-readable display string."""
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    size = float(n)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024


def sha256sum(path):
    """Hash in chunks to avoid loading 325 MB into memory at once."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# Download imagenette dataset
def download(url, dest):
    """Stream to a .part file, then atomically rename."""
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "image-diffusion/0.1"})

    print(f"  GET {url}")
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    filled = int(40 * done / total)
                    bar = "#" * filled + " " * (40 - filled)
                    status = f"{format_bytes(done)} / {format_bytes(total)}"
                    print("\r  [" + bar + "] " + status, end="", flush=True)
    print()

    # Check: if download expected and actual don't match, return IOError
    if total and done != total:
        tmp.unlink(missing_ok=True)
        raise OSError(f"Truncated download: got {done} of {total} bytes")

    # Atomic Rename: successful download, go ahead and replace temporary
    os.replace(tmp, dest)


# Extract
def _is_within(base, target):
    """True if `target` resolves to somewhere inside `base`."""
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def safe_extract(archive, dest):
    """Extract, refusing anything that could write outside `dest`."""
    dest.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive, "r:gz") as tar:
        for m in tar.getmembers():
            if m.name.startswith("/") or ".." in Path(m.name).parts:
                raise RuntimeError(f"Unsafe member path: {m.name!r}")
            if not (m.isfile() or m.isdir()):
                raise RuntimeError(f"Refusing non-regular member: {m.name!r}")
            if not _is_within(dest, dest / m.name):
                raise RuntimeError(f"Member escapes destination: {m.name!r}")

        if sys.version_info >= (3, 12):
            tar.extractall(path=dest, filter="data")
        else:
            tar.extractall(path=dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data", help="where to put the dataset")
    ap.add_argument("--force", action="store_true",
                    help="re-download and re-extract")
    args = ap.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / ARCHIVE_NAME  # .tgz file
    extract_root = root / EXTRACT_NAME  # Extracted file

    # 1. Acquire the .tgz file
    if archive.exists() and not args.force:
        print(f"tarball already present ({format_bytes(archive.stat().st_size)})")
    else:
        download(URL, archive)
        print(f"downloaded {format_bytes(archive.stat().st_size)}")

    # 2. Verify integrity using Secure Hash Algorithm (SHA), 256-bit
    digest = sha256sum(archive)
    if EXPECTED_SHA256 is None:
        print("\nEXPECTED_SHA256 is unset. Record this in the script:")
        print(f"EXPECTED_SHA256 = \"{digest}\"\n")
    elif digest != EXPECTED_SHA256:
        print(f"sha256 MISMATCH\n  got      {digest}"
              f"\n  expected {EXPECTED_SHA256}")
        print("Deleting the corrupt tarball. Re-run to retry.")
        archive.unlink()
        return 1
    else:  # Everything is good w.r.t. SHA256
        print(f"sha256 verified ({digest[:16]}...)")

    # 3. Extract .tgz file
    if extract_root.is_dir() and not args.force:
        print(f"already extracted at {extract_root}")
    else:
        print(f"extracting to {extract_root} ...")
        safe_extract(archive, root)
        print(f"extracted to {extract_root}")

    # 4. Hand off: Acquisition (not quality check) succeeded
    print(f"\nacquisition complete: {extract_root}")
    print(f"next: python scripts/verify_data.py --root {extract_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
