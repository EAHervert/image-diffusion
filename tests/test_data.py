"""Tests for the ImageNette data pipeline.
Assumes data has been downloaded.

Usage:
    pytest -q

Note: portions of this file were generated with the assistance of
Anthropic's Claude Opus 5, then reviewed line by line by the author.
"""

from pathlib import Path

import pytest
import torch

from image_diffusion.data import (
    _stratified_split_indices,
    build_imagenette_loader,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "imagenette2-320"

IMAGE_SIZE = 128
EXPECTED_SIZES = {"train": 9469, "val": 1959, "test": 1966}


@pytest.fixture
def targets():
    """All ten labels, uneven counts, none below 2."""
    counts = (6, 5, 4, 3, 2, 6, 5, 4, 3, 2)
    return [label for label, n in enumerate(counts) for _ in range(n)]


def test_split_is_disjoint_and_exhaustive(targets):
    a, b = _stratified_split_indices(targets, 0.5, seed=1234)
    assert set(a).isdisjoint(b)
    assert sorted(a + b) == list(range(len(targets)))


def test_split_is_deterministic(targets):
    assert (_stratified_split_indices(targets, 0.5, seed=1234)
            == _stratified_split_indices(targets, 0.5, seed=1234))


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_batch_split_sizes(split):
    loader = build_imagenette_loader(
        str(DATA_ROOT), split, batch_size=8, num_workers=0
    )
    assert len(loader.dataset) == EXPECTED_SIZES[split]


def test_batch_shape_dtype_and_range():
    loader = build_imagenette_loader(
        str(DATA_ROOT), "train", batch_size=4, num_workers=0
    )
    x, y = next(iter(loader))

    assert x.shape == (4, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert x.dtype == torch.float32
    assert torch.isfinite(x).all()
    assert x.min() >= -1.0 and x.max() <= 1.0
    assert x.abs().max() > 0.5  # a silently all-gray batch would fail here

    assert y.dtype == torch.long
    assert y.shape == (4,)
    assert int(y.min()) >= 0 and int(y.max()) < 10
