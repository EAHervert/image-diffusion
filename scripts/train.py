"""Training entry point for image-diffusion.

Usage:
    python scripts/train.py --config configs/base.yaml
    python scripts/train.py --config configs/base.yaml train.lr=2e-4
"""

import argparse
import torch
import torch.optim as optim
import time
import os
import math
from pathlib import Path
from omegaconf import OmegaConf
from torchvision.utils import make_grid, save_image

from image_diffusion import REGISTRY
from image_diffusion.data import build_imagenette_loader, denormalize
from image_diffusion.flow import sample_triple, flow_matching_loss
from image_diffusion.model import DiT



def main():
    # Parser for training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--config-data", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()

    # Load config and override with terminal arguments
    cfg = OmegaConf.load(args.config)

    if args.config_data:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.config_data))

    OmegaConf.set_struct(cfg, True)

    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    print(OmegaConf.to_yaml(cfg))

    for k in ("num_classes", "hflip", "root"):
        if k not in cfg.data:
            raise KeyError(f"cfg.data.{k} missing!")

    for k in ("steps", "lr", "seed", "grid_every"):
        if k not in cfg.train:
            raise KeyError(f"cfg.train.{k} missing!")

    # Device casting
    device = torch.device(cfg.train.device)

    # Random seed for everyone
    torch.manual_seed(cfg.train.seed)

    # Dataloader - training data
    dataloader = build_imagenette_loader(
        root=cfg.data.root, split="train", batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers, image_size=cfg.data.image_size, 
        hflip=cfg.data.hflip,
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

    warmup = max(1, int(cfg.train.warmup_frac * cfg.train.steps))
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-6, end_factor=1.0, 
        total_iters=warmup,)

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save_ckpt(tag):
        path = ckpt_dir / f"ckpt_{tag}.pt"
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": train_step,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        tmp = path.with_suffix(".tmp")
        torch.save(ckpt, tmp)
        os.replace(tmp, path)
        print(f"saved {path}")

    # Split the t values into N_Bins to evaluate differing levels of noise
    N_BINS = 5
    GRID_EVERY = int(cfg.train.grid_every)
    bin_sum = torch.zeros(N_BINS)
    bin_cnt = torch.zeros(N_BINS)
    BIN_LABELS = ["B1 [0.0-0.2]", "B2 [0.2-0.4]", "B3 [0.4-0.6]",
                "B4 [0.6-0.8]", "B5 [0.8-1.0]"]

    grid_dir = Path("docs/assets")
    grid_dir.mkdir(parents=True, exist_ok=True)

    # Fixed noise and labels, drawn once: every grid differs only by the weights
    _g = torch.Generator().manual_seed(0)
    y_grid = torch.arange(cfg.data.num_classes).repeat_interleave(2).to(device)
    x_grid = torch.randn(len(y_grid), 3, cfg.data.image_size, cfg.data.image_size, generator=_g).to(device)

    # CSV for loss curve
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / f"metrics_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    csv_file = csv_path.open("w", buffering=1)   # line-buffered: survives a crash
    csv_file.write("step,loss,lr,ms_per_step," + ",".join(f"b{i+1}" for i in range(N_BINS)) + "\n")

    print(f"logging metrics to {csv_path}\n")

    @torch.no_grad()
    def save_grid(tag):
        # Model switches to eval mode for generating image grids.
        model.eval()

        try:
            if cfg.sample.sampler not in REGISTRY:
                raise KeyError(f"Sampler '{cfg.sample.sampler}' not found.")

            sampler_fn = REGISTRY[cfg.sample.sampler]
            x_out = sampler_fn(model, x_grid, y_grid, cfg.sample.num_steps)
            out = grid_dir / f"samples_train_{tag}.png"
            save_image(make_grid(denormalize(x_out).cpu(), nrow=2), out)

            print(f"saved {out}\n")

        finally:
            # Switch back to training mode
            model.train()

    # Perform the training loop - manual count of the loops though the dataloader
    train_step = 0

    # Resume with previous weights if --resume is given
    if args.resume:
        print(f"Resuming - checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        train_step = checkpoint["step"]

        print(f"Resumed successfully at step {train_step}")

        if train_step >= int(cfg.train.steps):
            raise SystemExit(
                f"Checkpoint is at step {train_step}, train.steps is {cfg.train.steps}."
            )

    t_last = time.perf_counter()
    steps_at_t_last = train_step  # Track step count at last log/reset
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

            with torch.no_grad():
                per_sample = ((v_pred - v_target) ** 2).flatten(1).mean(1)
                idx = (t * N_BINS).long().clamp_(0, N_BINS - 1).cpu()
                bin_sum.index_add_(0, idx, per_sample.cpu())
                bin_cnt.index_add_(0, idx, torch.ones(len(idx)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Log the lr rate for monitoring
            if train_step % cfg.train.log_every == 0:
                if device.type == "mps":
                    torch.mps.synchronize()
                elif device.type == "cuda":
                    torch.cuda.synchronize()

                now = time.perf_counter()
                n = max(1, train_step - steps_at_t_last)
                ms = 1000.0 * (now - t_last) / n
                t_last = now
                steps_at_t_last = train_step

                lr = scheduler.get_last_lr()[0]
                means = (bin_sum / bin_cnt).tolist()
                bstr = "  ".join(f"{BIN_LABELS[i]}:{m:.3f}" for i, m in enumerate(means))
                print(f"step {train_step:>6d}  loss {loss.item():.4f}  "
                        f"lr {lr:.2e}  {ms:7.1f} ms/step  |  {bstr}")

                csv_file.write(
                    f"{train_step},{loss.item():.6f},{lr:.6e},{ms:.1f}," +
                    ",".join("" if math.isnan(m) else f"{m:.6f}" for m in means) + "\n")

                bin_sum.zero_()
                bin_cnt.zero_()

            train_step += 1

            if train_step % int(cfg.train.ckpt_every) == 0:
                save_ckpt(f"{train_step:06d}")

            if train_step % GRID_EVERY == 0:
                save_grid(f"{train_step:06d}")

                t_last = time.perf_counter()
                steps_at_t_last = train_step

            if train_step >= int(cfg.train.steps):
                break

    save_ckpt("last")
    save_grid("last")
    csv_file.close()


if __name__ == "__main__":
    main()
