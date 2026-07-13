# image-diffusion

Class-conditional DiT image generation using **flow matching**, built from the ground-up on PyTorch. 

## Status
v0.0 - Initial scaffold — no model code yet. 

## Install
    pip install -e .

## Layout
    src/image_diffusion/  # model, flow, samplers, data
    configs/              # YAML training configs
    scripts/              # train.py entry point
    tests/                # smoke tests

## License
MIT