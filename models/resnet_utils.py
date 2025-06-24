"""
Source: https://github.com/dotchen/LearningByCheating/blob/release-0.9.6/bird_view/models/common.py
"""

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import get_resnet


def select_branch(branches: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
    """
    Selects a specific branch from a stacked tensor of outputs using a command vector.

    Args:
        branches (Tensor): Shape (B, N, *), where N = num branches.
        command (Tensor): Shape (B), command vector indicating the selected branch for each sample.

    Returns:
        Tensor: Shape (B, *) for the selected branch per sample.
    """
    command = command.long()  # Ensure command is long tensor for indexing
    return branches[torch.arange(branches.size(0)), command]


class PixelToWorld(nn.Module):
    def __init__(self, steps):
        super().__init__()



class ResnetBase(nn.Module):
    def __init__(self, backbone, input_channel=3, bias_first=True, pretrained=False):
        super().__init__()
        

        conv, c = get_resnet(
                backbone, input_channel=input_channel,
                bias_first=bias_first, pretrained=pretrained)

        self.conv = conv
        self.c = c

        self.backbone = backbone
        self.input_channel = input_channel
        self.bias_first = bias_first


class Normalize(nn.Module):
    """
    Normalizes input images using provided mean and std.

    Input shape: (N, C, H, W)
    Output shape: (N, C, H, W)
    """
    def __init__(self, mean, std):
        super().__init__()
        self.mean = nn.Parameter(torch.FloatTensor(mean).reshape(1, 3, 1, 1), requires_grad=False)
        self.std = nn.Parameter(torch.FloatTensor(std).reshape(1, 3, 1, 1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class NormalizeV2(nn.Module):
    """
    CUDA-specific version of Normalize with fixed mean and std on GPU.

    Input shape: (N, C, H, W)
    Output shape: (N, C, H, W)
    """
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.FloatTensor(mean).reshape(1, 3, 1, 1)
        self.std = torch.FloatTensor(std).reshape(1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return (x - mean) / std


class SpatialSoftmax(nn.Module):
    """
    Computes expected 2D positions from a feature map using softmax attention.

    Input:
        feature (Tensor): shape (N, C, H, W) if data_format='NCHW'
                          or (N, H, W, C) if data_format='NHWC'

    Output:
        Tensor of shape (N, C, 2) containing (x, y) coordinates for each channel.

    Example:
        softmax = SpatialSoftmax(height=64, width=64, channel=32)
        out = softmax(torch.rand(8, 32, 64, 64))  # out.shape: (8, 32, 2)
    """
    def __init__(self, height: int, width: int, channel: int, temperature: float = None, data_format: str = 'NCHW'):
        super().__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        self.temperature = nn.Parameter(torch.ones(1) * temperature) if temperature else 1.0

        # NOTE: x is distance ahead, y is left/right
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self.height),
            np.linspace(-1., 1., self.width)
        )
        self.register_buffer('pos_x', torch.from_numpy(pos_x.reshape(-1)).float())
        self.register_buffer('pos_y', torch.from_numpy(pos_y.reshape(-1)).float())

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if self.data_format == 'NHWC':
            feature = feature.permute(0, 3, 1, 2)  # to NCHW
        N, C, H, W = feature.shape
        feature = feature.view(N * C, H * W)

        weight = F.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x * weight, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * weight, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], dim=1)

        return expected_xy.view(N, C, 2)


class SpatialSoftmaxBZ(nn.Module):
    def __init__(self, height: int, width: int):
        super().__init__()
        self.height = height
        self.width = width

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self.height),
            np.linspace(-1.0, 1.0, self.width)
        )
        self.pos_x = nn.Parameter(torch.from_numpy(pos_x.reshape(-1)).float(), requires_grad=False)
        self.pos_y = nn.Parameter(torch.from_numpy(pos_y.reshape(-1)).float(), requires_grad=False)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        N, C, H, W = feature.shape
        flattened = feature.view(N, C, -1)
        softmax = F.softmax(flattened, dim=-1)

        expected_x = torch.sum(self.pos_y * softmax, dim=-1)
        expected_x = (-expected_x + 1) / 2.0  # flip y-axis to [0, 1]

        expected_y = torch.sum(self.pos_x * softmax, dim=-1)
        expected_xy = torch.stack([expected_x, expected_y], dim=2)

        return expected_xy
