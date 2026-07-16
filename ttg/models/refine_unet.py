from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim > 1:
            t = t.view(-1)
        half_dim = self.dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim, device=t.device) * -scale)
        embeddings = t.float()[:, None] * frequencies[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


def create_norm(norm_type: str, num_channels: int) -> nn.Module:
    if norm_type == "gn":
        if num_channels <= 0:
            return nn.Identity()
        num_groups = 32 if num_channels >= 32 and num_channels % 32 == 0 else 1
        return nn.GroupNorm(num_groups, num_channels)
    raise TypeError(f"Unsupported norm type: {norm_type}")


def create_activation(activation_type: str) -> nn.Module:
    if activation_type == "silu":
        return nn.SiLU()
    if activation_type == "relu":
        return nn.ReLU()
    raise TypeError(f"Unsupported activation type: {activation_type}")


class ResBlockRefine(nn.Module):
    def __init__(self, in_c: int, out_c: int, time_c: int, norm_type: str = "gn", activation_type: str = "silu"):
        super().__init__()
        self.norm1 = create_norm(norm_type, in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.time_emb = nn.Sequential(create_activation(activation_type), nn.Linear(time_c, out_c))
        self.norm2 = create_norm(norm_type, out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.activation = create_activation(activation_type)
        self.residual_conv = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.norm1(x))
        out = self.conv1(out)
        out = out + self.time_emb(t_emb)[:, :, None, None]
        out = self.activation(self.norm2(out))
        out = self.conv2(out)
        return out + self.residual_conv(x)


class SelfAttentionBlockRefine(nn.Module):
    def __init__(self, num_channels: int, norm_type: str = "gn"):
        super().__init__()
        self.norm = create_norm(norm_type, num_channels)
        self.qkv = nn.Conv2d(num_channels, num_channels * 3, 1)
        self.out = nn.Conv2d(num_channels, num_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        norm_x = self.norm(x)
        q, k, v = self.qkv(norm_x).chunk(3, dim=1)
        q = q.view(n, c, h * w).permute(0, 2, 1)
        k = k.view(n, c, h * w)
        v = v.view(n, c, h * w)
        attn = F.softmax(torch.bmm(q, k) * (c ** -0.5), dim=-1)
        res = torch.bmm(v, attn.permute(0, 2, 1)).view(n, c, h, w)
        return x + self.out(res)


class ResAttnBlockRefine(nn.Module):
    def __init__(self, in_c: int, out_c: int, time_c: int, with_attn: bool, norm_type: str = "gn", activation_type: str = "silu"):
        super().__init__()
        self.res_block = ResBlockRefine(in_c, out_c, time_c, norm_type, activation_type)
        self.attn_block = SelfAttentionBlockRefine(out_c, norm_type) if with_attn else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        return self.attn_block(self.res_block(x, t_emb))


class UNetRefine(nn.Module):
    """U-Net diffusion backbone used in the TTG refinement phase."""

    def __init__(
        self,
        img_shape: tuple[int, int],
        channels: list[int] | tuple[int, ...],
        in_channels: int,
        out_channels: int,
        pe_dim: int,
        with_attns: list[bool] | tuple[bool, ...] | None = None,
        norm_type: str = "gn",
        activation_type: str = "silu",
    ) -> None:
        super().__init__()
        _ = img_shape
        channels = list(channels)
        if with_attns is None:
            with_attns = [False] * len(channels)
        if len(with_attns) != len(channels):
            raise ValueError("with_attns must have the same length as channels")

        time_c = pe_dim * 4
        self.time_embedding = nn.Sequential(
            SinusoidalPositionalEmbedding(pe_dim),
            nn.Linear(pe_dim, time_c),
            nn.GELU(),
            nn.Linear(time_c, time_c),
        )
        self.conv_in = nn.Conv2d(in_channels, channels[0], 3, 1, 1)

        self.downs = nn.ModuleList()
        self.down_samplers = nn.ModuleList()
        current_c = channels[0]
        for i, (channel, with_attn) in enumerate(zip(channels, with_attns)):
            self.downs.append(ResAttnBlockRefine(current_c, channel, time_c, with_attn, norm_type, activation_type))
            self.down_samplers.append(nn.Conv2d(channel, channel, 3, 2, 1) if i != len(channels) - 1 else nn.Identity())
            current_c = channel

        self.mid = ResAttnBlockRefine(current_c, current_c, time_c, True, norm_type, activation_type)

        self.ups = nn.ModuleList()
        self.up_samplers = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        reversed_with_attns = list(reversed(with_attns))
        for i, (channel, with_attn) in enumerate(zip(reversed_channels, reversed_with_attns)):
            in_c_cat = current_c + channel
            self.ups.append(ResAttnBlockRefine(in_c_cat, channel, time_c, with_attn, norm_type, activation_type))
            out_c_for_upsampler = reversed_channels[i + 1] if i < len(channels) - 1 else channels[0]
            self.up_samplers.append(nn.ConvTranspose2d(channel, out_c_for_upsampler, 4, 2, 1) if i != len(channels) - 1 else nn.Identity())
            current_c = out_c_for_upsampler

        self.conv_out = nn.Sequential(
            create_norm(norm_type, channels[0]),
            create_activation(activation_type),
            nn.Conv2d(channels[0], out_channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embedding(t)
        x = torch.cat((x, condition), dim=1)
        x = self.conv_in(x)
        residuals = []
        for block, downsampler in zip(self.downs, self.down_samplers):
            x = block(x, t_emb)
            residuals.append(x)
            x = downsampler(x)
        x = self.mid(x, t_emb)
        for block, upsampler in zip(self.ups, self.up_samplers):
            x = torch.cat((x, residuals.pop()), dim=1)
            x = block(x, t_emb)
            x = upsampler(x)
        return self.conv_out(x)
