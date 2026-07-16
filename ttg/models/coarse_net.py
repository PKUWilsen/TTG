from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import init


def init_weights(net: nn.Module, init_type: str = "normal", gain: float = 0.02) -> None:
    def init_func(module: nn.Module) -> None:
        classname = module.__class__.__name__
        if hasattr(module, "weight") and ("Conv" in classname or "Linear" in classname):
            if init_type == "normal":
                init.normal_(module.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(module.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(module.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(module.weight.data, gain=gain)
            else:
                raise NotImplementedError(f"Unsupported initialization: {init_type}")
            if getattr(module, "bias", None) is not None:
                init.constant_(module.bias.data, 0.0)
        elif "BatchNorm2d" in classname:
            init.normal_(module.weight.data, 1.0, gain)
            init.constant_(module.bias.data, 0.0)
    net.apply(init_func)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, conv_transpose: bool = True) -> None:
        super().__init__()
        if conv_transpose:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class AttentionBlock(nn.Module):
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        psi = self.relu(self.gate_proj(gate) + self.skip_proj(skip))
        psi = self.psi(psi)
        return skip * psi


class AttUNet(nn.Module):
    """Attention U-Net used in the TTG coarse estimation phase."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        channel_list: list[int] | tuple[int, ...] = (64, 128, 256, 512, 1024),
        checkpoint: bool = False,
        conv_transpose: bool = True,
    ) -> None:
        super().__init__()
        if len(channel_list) < 5:
            raise ValueError("channel_list must have at least five elements.")
        c = list(channel_list)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = ConvBlock(in_channels, c[0])
        self.conv2 = ConvBlock(c[0], c[1])
        self.conv3 = ConvBlock(c[1], c[2])
        self.conv4 = ConvBlock(c[2], c[3])
        self.conv5 = ConvBlock(c[3], c[4])

        self.up5 = UpConv(c[4], c[3], conv_transpose)
        self.att5 = AttentionBlock(c[3], c[3], c[2])
        self.up_conv5 = ConvBlock(c[3] * 2, c[3])

        self.up4 = UpConv(c[3], c[2], conv_transpose)
        self.att4 = AttentionBlock(c[2], c[2], c[1])
        self.up_conv4 = ConvBlock(c[2] * 2, c[2])

        self.up3 = UpConv(c[2], c[1], conv_transpose)
        self.att3 = AttentionBlock(c[1], c[1], c[0])
        self.up_conv3 = ConvBlock(c[1] * 2, c[1])

        self.up2 = UpConv(c[1], c[0], conv_transpose)
        self.att2 = AttentionBlock(c[0], c[0], max(c[0] // 2, 1))
        self.up_conv2 = ConvBlock(c[0] * 2, c[0])

        self.out_conv = nn.Conv2d(c[0], out_channels, kernel_size=1)
        self.final_activation = nn.Tanh()
        if not checkpoint:
            init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x4 = self.conv4(self.pool(x3))
        x5 = self.conv5(self.pool(x4))

        d5 = self.up5(x5)
        d5 = torch.cat((self.att5(d5, x4), d5), dim=1)
        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat((self.att4(d4, x3), d4), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((self.att3(d3, x2), d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((self.att2(d2, x1), d2), dim=1)
        d2 = self.up_conv2(d2)
        return self.final_activation(self.out_conv(d2))
