#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ttg.datasets import build_dataloader
from ttg.diffusion import DDIMSampler
from ttg.models import AttUNet, UNetRefine
from ttg.utils import load_config, save_config, set_seed
from ttg.utils.metrics import calculate_metrics, save_metrics
from ttg.utils.model_utils import configure_logging, load_state_dict, log_model_parameters
from ttg.utils.sparse_sampling import create_enhanced_sparse_map_sift
from ttg.utils.visualization import visualize_comparison

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the TTG refinement phase.")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--coarse_ckpt", type=str, required=True)
    parser.add_argument("--refine_ckpt", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="outputs/eval")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def build_models(cfg, device, coarse_ckpt, refine_ckpt):
    coarse_cfg = cfg["coarse_model"]
    coarse = AttUNet(
        in_channels=int(coarse_cfg.get("in_channels", 2)),
        out_channels=int(coarse_cfg.get("out_channels", 1)),
        channel_list=coarse_cfg.get("channel_list", [128, 128, 256, 256, 512, 512]),
        checkpoint=True,
        conv_transpose=bool(coarse_cfg.get("conv_transpose", True)),
    ).to(device)
    load_state_dict(coarse, coarse_ckpt, device)
    coarse.eval()
    for param in coarse.parameters():
        param.requires_grad = False

    data_cfg = cfg["data"]
    refine_cfg = dict(cfg["model"])
    refine_cfg.pop("name", None)
    refine_cfg.pop("lr", None)
    refine = UNetRefine(
        img_shape=(int(data_cfg.get("img_height", 256)), int(data_cfg.get("img_width", 256))),
        **refine_cfg,
    ).to(device)
    load_state_dict(refine, refine_ckpt, device)
    refine.eval()
    for param in refine.parameters():
        param.requires_grad = False
    return coarse, refine


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(save_dir / "eval.log")
    cfg = load_config(args.config)
    if args.data_root is not None:
        cfg.setdefault("data", {})["data_root"] = args.data_root
    save_config(cfg, save_dir / "config_used.yaml")
    if args.seed is not None:
        set_seed(args.seed, deterministic=bool(cfg.get("deterministic", False)))
    device = torch.device(args.device)

    coarse_estimator, refine_model = build_models(cfg, device, args.coarse_ckpt, args.refine_ckpt)
    log_model_parameters(coarse_estimator, "Coarse estimator")
    log_model_parameters(refine_model, "Refinement model")

    eval_cfg = cfg["evaluation"]
    if int(eval_cfg.get("batch_size", 1)) != 1:
        logger.warning("The current evaluation pipeline assumes batch_size=1 for sparse enhancement and map election.")
    test_loader = build_dataloader(cfg, "test", batch_size=int(eval_cfg.get("batch_size", 1)), data_root=args.data_root, shuffle=True)
    diffusion_cfg = cfg["diffusion"]
    sampler = DDIMSampler(
        device,
        n_steps=int(diffusion_cfg["ddpm_steps"]),
        beta_start=float(diffusion_cfg.get("beta_start", 1e-4)),
        beta_end=float(diffusion_cfg.get("beta_end", 0.02)),
    )

    image_size = int(cfg["data"].get("img_height", 256))
    num_true_points = int(cfg["data"]["num_samples_sparse"])
    total_points = int(cfg.get("sparse_enhancement", {}).get("total_points", 300))
    ddim_steps = int(diffusion_cfg.get("ddim_steps", 50))
    use_sparse_election = bool(eval_cfg.get("use_sparse_election", True))

    predictions = []
    targets = []
    with torch.no_grad():
        for idx, (sparse_map, dense_map, building_map, _, _) in enumerate(tqdm(test_loader, desc="Evaluating")):
            if sparse_map.numel() == 0:
                continue
            if sparse_map.shape[0] != 1:
                raise ValueError("Evaluation currently expects batch_size=1.")
            target = dense_map.to(device)
            sparse = sparse_map.to(device)
            building = building_map.to(device)
            coarse_map = coarse_estimator(torch.cat([building, sparse], dim=1))

            enhanced_np = create_enhanced_sparse_map_sift(
                sparse[0, 0].detach().cpu().numpy(),
                coarse_map,
                num_true_points=num_true_points,
                num_total_points=total_points,
                image_size=image_size,
            )
            enhanced_cond = torch.from_numpy(np.stack([building[0, 0].detach().cpu().numpy(), enhanced_np])).unsqueeze(0).to(device=device, dtype=torch.float32)
            initial_noise = torch.randn((1, 1, image_size, image_size), device=device)
            refined_map = sampler.sample_reverse_three_stage_guidance(
                model=refine_model,
                x_T=initial_noise,
                coarse_map=coarse_map,
                enhanced_cond=enhanced_cond,
                sparse_map_cond=sparse,
                ddim_steps=ddim_steps,
                guidance_cfg=cfg["guidance"],
                show_progress=bool(eval_cfg.get("show_sampling_progress", True)),
            )

            final_map = refined_map
            if use_sparse_election:
                truth_mask = sparse[0, 0] != -1.0
                if truth_mask.sum() > 0:
                    coarse_err = F.mse_loss(coarse_map[0, 0][truth_mask], target[0, 0][truth_mask])
                    refined_err = F.mse_loss(refined_map[0, 0][truth_mask], target[0, 0][truth_mask])
                    final_map = coarse_map if coarse_err < refined_err else refined_map

            predictions.append(final_map.detach().cpu())
            targets.append(target.detach().cpu())
            if idx < int(eval_cfg.get("num_samples_to_visualize", 0)):
                visualize_comparison(coarse_map, refined_map, final_map, target, idx, save_dir)

    metrics = calculate_metrics(predictions, targets)
    metrics_path = save_metrics(metrics, save_dir, prefix="eval_metrics")
    logger.info("Evaluation metrics: %s", metrics)
    logger.info("Saved metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
