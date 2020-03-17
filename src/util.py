import torch
import torch.nn as nn


def conv(in_channels: int,
         out_channels: int,
         kernel_size: int,
         bias: bool = True,
         rate: int = 1,
         stride: int = 1) -> nn.Conv2d:
    padding = kernel_size // 2 if rate == 1 else rate
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, stride=stride, dilation=rate,
        padding=padding, bias=bias)


def tensor_round(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x - 0.001)
