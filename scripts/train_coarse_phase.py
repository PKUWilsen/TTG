#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
import time
import torch
import torch.nn as nn
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ttg.datasets import build_dataloader
from ttg.models import AttUNet
from ttg.utils import load_config, save_config, set_seed
from ttg.utils.model_utils import configure_logging, load_state_dict, log_model_parameters, save_checkpoint
from ttg.utils.visualization import plot_loss_curves

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the TTG coarse estimation phase.")
    parser.add_argument("--config", type=str, default="configs/coarse_phase.yaml")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="outputs/coarse_phase")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint to initialize the coarse estimator.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


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
    model_cfg = cfg["model"]
    model = AttUNet(
        in_channels=int(model_cfg.get("in_channels", 2)),
        out_channels=int(model_cfg.get("out_channels", 1)),
        channel_list=model_cfg.get("channel_list", [128, 128, 256, 256, 512, 512]),
        checkpoint=bool(args.resume),
        conv_transpose=bool(model_cfg.get("conv_transpose", True)),
    ).to(device)
    if args.resume:
        load_state_dict(model, args.resume, device)
    log_model_parameters(model, "Coarse estimator")

    train_cfg = cfg["training"]
    train_loader = build_dataloader(cfg, "train", batch_size=int(train_cfg["batch_size"]), data_root=args.data_root, shuffle=True)
    val_loader = build_dataloader(cfg, "val", batch_size=int(train_cfg.get("val_batch_size", train_cfg["batch_size"])), data_root=args.data_root, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 1e-7)),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(train_cfg.get("milestones", [50, 120])),
        gamma=float(train_cfg.get("gamma", 0.5)),
    )

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    num_epochs = int(train_cfg["num_epochs"])
    for epoch in range(num_epochs):
        start = time.time()
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [train]", leave=False)
        for sparse_map, dense_map, building_map, _, _ in pbar:
            if sparse_map.numel() == 0:
                continue
            condition = torch.cat([building_map, sparse_map], dim=1).to(device)
            target = dense_map.to(device)
            pred = model(condition)
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_size = target.shape[0]
            train_loss_sum += loss.item() * batch_size
            train_count += batch_size
            pbar.set_postfix(loss=f"{loss.item():.6f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
        train_loss = train_loss_sum / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0.0
        history["train_loss"].append(train_loss)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for sparse_map, dense_map, building_map, _, _ in val_loader:
                if sparse_map.numel() == 0:
                    continue
                condition = torch.cat([building_map, sparse_map], dim=1).to(device)
                target = dense_map.to(device)
                pred = model(condition)
                loss = criterion(pred, target)
                batch_size = target.shape[0]
                val_loss_sum += loss.item() * batch_size
                val_count += batch_size
        val_loss = val_loss_sum / len(val_loader.dataset) if len(val_loader.dataset) > 0 else float("inf")
        history["val_loss"].append(val_loss)
        scheduler.step()

        logger.info(
            "Epoch %d/%d | train %.6f | val %.6f | %.2fs",
            epoch + 1,
            num_epochs,
            train_loss,
            val_loss,
            time.time() - start,
        )
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, save_dir / "coarse_best.pth")
            logger.info("Saved best coarse estimator: val %.6f", best_val)

        if (epoch + 1) % int(train_cfg.get("plot_every_n_epochs", 10)) == 0:
            plot_loss_curves(history, save_dir / "loss_curve.png")

    plot_loss_curves(history, save_dir / "loss_curve_final.png")
    logger.info("Training finished. Best validation loss: %.6f", best_val)


if __name__ == "__main__":
    main()
