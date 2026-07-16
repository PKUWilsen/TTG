from __future__ import annotations

import torch
from tqdm import tqdm


class DDPM:
    """Basic DDPM scheduler and sampler retained for baseline or ablation use."""

    def __init__(self, device: str | torch.device, n_steps: int, min_beta: float = 1e-4, max_beta: float = 0.02) -> None:
        device = torch.device(device)
        betas = torch.linspace(min_beta, max_beta, n_steps, device=device)
        alphas = 1 - betas
        self.betas = betas
        self.n_steps = int(n_steps)
        self.alphas = alphas
        self.alpha_bars = torch.cumprod(alphas, dim=0)

    def sample_forward(self, x: torch.Tensor, t: torch.Tensor, eps: torch.Tensor | None = None) -> torch.Tensor:
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        if eps is None:
            eps = torch.randn_like(x)
        return eps * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * x

    def sample_backward(self, img_or_shape, net: torch.nn.Module, device: str | torch.device, simple_var: bool = True) -> torch.Tensor:
        device = torch.device(device)
        x = img_or_shape if isinstance(img_or_shape, torch.Tensor) else torch.randn(img_or_shape, device=device)
        net = net.to(device)
        for t in tqdm(range(self.n_steps - 1, -1, -1), desc="DDPM sampling"):
            x = self.sample_backward_step(x, t, net, simple_var)
        return x

    def sample_backward_step(self, x_t: torch.Tensor, t: int, net: torch.nn.Module, simple_var: bool = True) -> torch.Tensor:
        n = x_t.shape[0]
        t_tensor = torch.full((n,), t, dtype=torch.long, device=x_t.device)
        eps = net(x_t, t_tensor)
        if t == 0:
            noise = 0
        else:
            var = self.betas[t] if simple_var else (1 - self.alpha_bars[t - 1]) / (1 - self.alpha_bars[t]) * self.betas[t]
            noise = torch.randn_like(x_t) * torch.sqrt(var)
        mean = (x_t - (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) * eps) / torch.sqrt(self.alphas[t])
        return mean + noise
