"""Training entry point for image-diffusion.

Usage:
    python scripts/train.py --config configs/base.yaml
    python scripts/train.py --config configs/base.yaml train.lr=2e-4
"""

import argparse
import torch
import torch.optim as optim
from omegaconf import OmegaConf

from image_diffusion.data import build_imagenette_loader
from image_diffusion.flow import sample_triple, flow_matching_loss
from image_diffusion.model import DiT



def main():
    # Parser for training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    # Load config and override with terminal arguments
    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    print(OmegaConf.to_yaml(cfg))  # print resolved config so runs are self-documenting

    # Device casting
    device = torch.device(cfg.train.device)

    # Dataloader - training data
    dataloader = build_imagenette_loader(
        root=cfg.data.root, split="train", batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers, image_size=cfg.data.image_size
        )

    # DiT model
    model = DiT(
        image_size=(cfg.data.image_size, cfg.data.image_size),
        d_model=cfg.model.embed_dim, depth=cfg.model.depth,
        num_heads=cfg.model.num_heads, patch_size=cfg.model.patch_size,
        num_classes=cfg.model.num_classes
        ).to(device)

    model.train()  # Set model to training mode

    # Set optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
        betas=tuple(cfg.train.betas), eps=cfg.train.eps,)

    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-6, end_factor=1.0,
        total_iters=cfg.train.warmup_steps,)

    # Perform the training loop - manual count of the loops though the dataloader
    train_step = 0
    while train_step < int(cfg.train.steps):
        for x_1, y in dataloader:
            # Move batch to device
            x_1 = x_1.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Get the interpolated image, time step(s), and velocity
            x_t, t, v_target = sample_triple(x_1)
            v_pred = model(x_t, t, y)

            # Calculate loss and adjust weights based on said loss
            loss = flow_matching_loss(v_pred, v_target)  # MSE loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Log the lr rate for monitoring
            if train_step % cfg.train.log_every == 0:
                lr = scheduler.get_last_lr()[0]
            
            print(f"step {train_step:>6d}  loss {loss.item():.4f}  lr {lr:.2e}")

            train_step += 1


if __name__ == "__main__":
    main()
