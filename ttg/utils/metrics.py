from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Dict, Iterable
import numpy as np
import torch
import torch.nn as nn


def calculate_metrics(predictions: Iterable[torch.Tensor], targets: Iterable[torch.Tensor]) -> Dict[str, float]:
    pred = torch.cat(list(predictions), dim=0).detach().cpu().float()
    target = torch.cat(list(targets), dim=0).detach().cpu().float()
    pred_01 = (pred + 1.0) / 2.0
    target_01 = (target + 1.0) / 2.0
    mse = nn.MSELoss()(pred_01, target_01).item()
    rmse = float(np.sqrt(mse))
    psnr = float(20.0 * np.log10(1.0 / rmse)) if rmse > 0 else 100.0
    error_sum_sq = torch.sum((pred_01 - target_01) ** 2).item()
    target_power = torch.sum(target_01 ** 2).item()
    nmse = float(error_sum_sq / target_power) if target_power > 1e-8 else float("inf")
    return {"MSE": float(mse), "RMSE": rmse, "PSNR": psnr, "NMSE": nmse}


def save_metrics(metrics: Dict[str, float], save_dir: str | Path, prefix: str = "metrics") -> Path:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = save_dir / f"{prefix}_{timestamp}.txt"
    with path.open("w", encoding="utf-8") as f:
        f.write("TTG Evaluation Report\n")
        f.write("=" * 40 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.6f}\n")
    return path
