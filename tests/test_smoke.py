"""
Smoke tests: version check + submodule imports.
"""

import image_diffusion


# Test for image_diffusion version
def test_version():
    assert image_diffusion.__version__ == "0.0.0"


# Test to see if submodules are being imported
def test_submodule_imports():
    from image_diffusion import data, flow, model, samplers
