#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ttg.datasets import build_dataloader
from ttg.diffusion import DDIMSampler
from ttg.models import AttUNet, UNetRefine
from ttg.utils import load_config, save_config, set_seed
from ttg.utils.model_utils import configure_logging, load_state_dict, log_model_parameters, save_checkpoint
from ttg.utils.sparse_sampling import create_enhanced_sparse_map_sift

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the TTG refinement phase.")
    parser.add_argument("--config", type=str, default="configs/refine_phase.yaml")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--coarse_ckpt", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="outputs/refine_phase")
    parser.add_argument("--resume", type=str, default=None, help="Optional refinement checkpoint to initialize the model.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def build_coarse_estimator(cfg, device, checkpoint_path):
    model_cfg = cfg["coarse_model"]
    model = AttUNet(
        in_channels=int(model_cfg.get("in_channels", 2)),
        out_channels=int(model_cfg.get("out_channels", 1)),
        channel_list=model_cfg.get("channel_list", [128, 128, 256, 256, 512, 512]),
        checkpoint=True,
        conv_transpose=bool(model_cfg.get("conv_transpose", True)),
    ).to(device)
    load_state_dict(model, checkpoint_path, device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def build_refine_model(cfg, device, resume=None):
    data_cfg = cfg["data"]
    model_cfg = dict(cfg["model"])
    model_cfg.pop("name", None)
    model_cfg.pop("lr", None)
    model = UNetRefine(
        img_shape=(int(data_cfg.get("img_height", 256)), int(data_cfg.get("img_width", 256))),
        **model_cfg,
    ).to(device)
    if resume:
        load_state_dict(model, resume, device)
    return model


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(save_dir / "train.log")
    cfg = load_config(args.config)
    if args.data_root is not None:
        cfg.setdefault("data", {})["data_root"] = args.data_root
    save_config(cfg, save_dir / "config_used.yaml")
    if args.seed is not None:
        set_seed(args.seed, deterministic=bool(cfg.get("deterministic", False)))
    device = torch.device(args.device)

    coarse_estimator = build_coarse_estimator(cfg, device, args.coarse_ckpt)
    log_model_parameters(coarse_estimator, "Coarse estimator")

    model = build_refine_model(cfg, device, args.resume)
    log_model_parameters(model, "Refinement model")

    diffusion_cfg = cfg["diffusion"]
    ddpm = DDIMSampler(
        device,
        n_steps=int(diffusion_cfg["ddpm_steps"]),
        beta_start=float(diffusion_cfg.get("beta_start", 1e-4)),
        beta_end=float(diffusion_cfg.get("beta_end", 0.02)),
    )
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    train_loader = build_dataloader(cfg, "train", batch_size=int(train_cfg["batch_size"]), data_root=args.data_root, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg.get("weight_decay", 0.0)))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(train_cfg.get("scheduler_step_size", 200)),
        gamma=float(train_cfg.get("scheduler_gamma", 0.5)),
    )
    loss_fn = nn.MSELoss()
    best_loss = float("inf")

    log_path = save_dir / "train_metrics.csv"
    log_path.write_text("epoch,average_loss,learning_rate,timestamp\n", encoding="utf-8")

    image_size = int(data_cfg.get("img_height", 256))
    num_true_points = int(data_cfg["num_samples_sparse"])
    total_points = int(cfg.get("sparse_enhancement", {}).get("total_points", 300))

    num_epochs = int(train_cfg["num_epochs"])
    for epoch in range(num_epochs):
        model.train()
        loss_sum = 0.0
        batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for sparse_map, dense_map, building_map, _, _ in pbar:
            if sparse_map.numel() == 0:
                continue
            target = dense_map.to(device)
            building = building_map.to(device)
            sparse = sparse_map.to(device)
            batch_size = target.shape[0]

            with torch.no_grad():
                coarse_map = coarse_estimator(torch.cat([building, sparse], dim=1))
                enhanced_maps = []
                for i in range(batch_size):
                    enhanced = create_enhanced_sparse_map_sift(
                        sparse[i, 0].detach().cpu().numpy(),
                        coarse_map[i].unsqueeze(0),
                        num_true_points=num_true_points,
                        num_total_points=total_points,
                        image_size=image_size,
                    )
                    enhanced_maps.append(enhanced)
                enhanced_sparse = torch.from_numpy(np.stack(enhanced_maps)).unsqueeze(1).to(device=device, dtype=torch.float32)

            t = torch.randint(0, int(diffusion_cfg["ddpm_steps"]), (batch_size,), device=device)
            noise = torch.randn_like(target)
            x_t = ddpm.sample_forward(target, t, noise)
            condition = torch.cat([building, enhanced_sparse], dim=1)
            predicted_noise = model(x_t, t, condition)
            loss = loss_fn(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            batches += 1
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        scheduler.step()
        avg_loss = loss_sum / len(train_loader) if len(train_loader) > 0 else 0.0
        lr = scheduler.get_last_lr()[0]
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"{epoch + 1},{avg_loss:.8f},{lr:.8e},{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        logger.info("Epoch %d/%d | loss %.6f | lr %.2e", epoch + 1, num_epochs, avg_loss, lr)
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, save_dir / "refine_best.pth")
            logger.info("Saved best refinement model: loss %.6f", best_loss)

    logger.info("Refinement-phase training finished. Best loss: %.6f", best_loss)


if __name__ == "__main__":
    main()
