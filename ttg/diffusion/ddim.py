from __future__ import annotations

from typing import Dict, Any
import torch
from tqdm import tqdm


class DDIMSampler:
    """DDIM sampler with three-stage guidance for the TTG refinement phase."""

    def __init__(self, device: str | torch.device, n_steps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02) -> None:
        self.device = torch.device(device)
        self.n_steps = int(n_steps)
        self.betas = torch.linspace(beta_start, beta_end, self.n_steps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim=0)

    def sample_forward(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_0, device=self.device)
        sqrt_alpha_hat = self.alpha_hats.gather(0, t).sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_hat = (1 - self.alpha_hats.gather(0, t)).sqrt().view(-1, 1, 1, 1)
        return sqrt_alpha_hat * x_0 + sqrt_one_minus_alpha_hat * noise

    def _get_x0_pred_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, eps_theta: torch.Tensor) -> torch.Tensor:
        alpha_hat_t = self.alpha_hats.gather(0, t).view(-1, 1, 1, 1)
        x0_pred = (x_t - (1 - alpha_hat_t).sqrt() * eps_theta) / (alpha_hat_t.sqrt() + 1e-8)
        return x0_pred.clamp(-1, 1)

    def _get_x_prev_from_x0_pred(self, t_next: torch.Tensor, x0_pred: torch.Tensor, eps_theta: torch.Tensor) -> torch.Tensor:
        alpha_hat_next = self.alpha_hats.gather(0, t_next).view(-1, 1, 1, 1)
        return alpha_hat_next.sqrt() * x0_pred + (1 - alpha_hat_next).sqrt() * eps_theta

    @torch.no_grad()
    def sample_reverse_three_stage_guidance(
        self,
        model: torch.nn.Module,
        x_T: torch.Tensor,
        coarse_map: torch.Tensor,
        enhanced_cond: torch.Tensor,
        sparse_map_cond: torch.Tensor,
        ddim_steps: int,
        guidance_cfg: Dict[str, Any],
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Reverse DDIM sampling with three-stage guidance.

        Stage I injects high-confidence core regions from the coarse map.
        Stage II lets the diffusion model generate freely.
        Stage III enforces sparse-point consistency by noising known sparse values
        to the next timestep and replacing the corresponding positions.
        """
        x = x_T
        batch_size = x_T.shape[0]
        step_indices = torch.linspace(self.n_steps - 1, 0, int(ddim_steps), dtype=torch.long, device=self.device)

        core_threshold = float(guidance_cfg.get("core_threshold_01", 0.9))
        stage1_end_step = int(guidance_cfg["stage1_end_step"])
        stage2_end_step = int(guidance_cfg["stage2_end_step"])

        core_mask = (coarse_map > (core_threshold * 2.0 - 1.0)).float()
        core_values = coarse_map * core_mask
        truth_mask = (sparse_map_cond != -1.0).float()

        iterator = range(len(step_indices) - 1)
        if show_progress:
            iterator = tqdm(iterator, desc="Three-stage DDIM sampling", ncols=100)

        for i in iterator:
            t = torch.full((batch_size,), step_indices[i], device=self.device, dtype=torch.long)
            t_next = torch.full((batch_size,), step_indices[i + 1], device=self.device, dtype=torch.long)
            predicted_noise = model(x, t, enhanced_cond)
            x0_pred = self._get_x0_pred_from_noise(x, t, predicted_noise)
            x_pred = self._get_x_prev_from_x0_pred(t_next, x0_pred, predicted_noise)

            current_step = int(t[0].item())
            if current_step >= stage1_end_step:
                # Stage I: coarse-core injection.
                x = x_pred * (1 - core_mask) + core_values * core_mask
            elif current_step < stage2_end_step:
                # Stage III: sparse-point consistency.
                truth_at_t_next = self.sample_forward(sparse_map_cond, t_next)
                x = x_pred * (1 - truth_mask) + truth_at_t_next * truth_mask
            else:
                # Stage II: free generation.
                x = x_pred
        return x

    # Backward-compatible alias for the original scripts.
    sample_reverse_staged_guidance = sample_reverse_three_stage_guidance
