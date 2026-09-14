"""
Devices will be different depending on where they are running:

    Local - cpu: true, mps: true, cuda: false
    Cloud - cpu: true, mps: false, cuda: true

"""

import pytest
import torch

from image_diffusion import get_device


@pytest.mark.parametrize("name, available", [
    ("cuda", torch.cuda.is_available()),
    ("mps", torch.backends.mps.is_available()),
    ("cpu", True)
])
def test_explicit_matches_availability(name, available):
    if available:
        assert get_device(name).type == name
    else:
        with pytest.raises(RuntimeError):
            get_device(name)

# Option 'auto; hould return ANY valid device
def test_auto():
    device = get_device(target='auto')

    assert isinstance(device, torch.device)

# Passing an invalid name should raise a ValueError
def test_invalid_name_raises():
    with pytest.raises(ValueError):
        get_device("gpu")