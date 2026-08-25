"""Sample a class-conditional grid from a trained checkpoint.

Usage:
    python scripts/sample.py --ckpt checkpoints/ckpt_last.pt

Note: portions of this file were generated with the assistance of
Anthropic's Claude Opus 5, then reviewed line by line by the author.
"""
import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torchvision.utils import make_grid, save_image

from image_diffusion import REGISTRY
from image_diffusion.data import denormalize
from image_diffusion.model import DiT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/ckpt_last.pt")
    parser.add_argument("--out", type=str, default="docs/assets/samples.png")
    parser.add_argument("--per-class", type=int, default=4)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    device = torch.device(cfg.train.device)

    model = DiT(
        image_size=(cfg.data.image_size, cfg.data.image_size),
        d_model=cfg.model.embed_dim, depth=cfg.model.depth,
        num_heads=cfg.model.num_heads, patch_size=cfg.model.patch_size,
        num_classes=cfg.model.num_classes,
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    k = args.per_class
    y = torch.arange(cfg.data.num_classes, device=device).repeat_interleave(k)
    x = torch.randn(len(y), 3, cfg.data.image_size, cfg.data.image_size, device=device)

    sampler = REGISTRY[cfg.sample.sampler]
    x_1 = sampler(model, x, y, cfg.sample.num_steps)
    grid = make_grid(denormalize(x_1).cpu(), nrow=k)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    save_image(grid, out)

    print(f"saved {out}  step={ckpt['step']}  num_steps={cfg.sample.num_steps}")


if __name__ == "__main__":
    main()
