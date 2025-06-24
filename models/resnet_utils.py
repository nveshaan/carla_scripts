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


class CoordConverter:
    """
    Converts normalized 2D image coordinates into bird’s-eye view (BEV) coordinates
    in a top-down map, considering camera intrinsic parameters and pitch.

    Args:
        w (int): Width of the input image in pixels.
        h (int): Height of the input image in pixels.
        fov (float): Horizontal field of view (degrees).
        world_y (float): Height of the camera above ground plane in meters.
        pitch (float): Camera pitch in degrees. Negative means looking downward.
        fixed_offset (float): Shift in pixels to push ego-agent downward in the map.
        pixels_per_meter (float): Resolution of map (pixels per meter).
        crop_size (int): Size of the square top-down map in pixels.
        device (str): Torch device (e.g. 'cuda' or 'cpu').

    Source: https://github.com/dotchen/LearningByCheating/blob/4145d33f74c9a8f27061a0f94840f3e458ecc60e/training/train_image_phase1.py#L35
    """
    # TODO: transfer these values to config, don't hard code
    def __init__(self, w=320, h=240, fov=90.0, world_y=2.0, pitch=-10.0, fixed_offset=2.0,
                 pixels_per_meter=5.0, crop_size=192, device='cuda'):
        self._img_size = torch.FloatTensor([w, h]).to(device)
        self._fov = fov
        self._world_y = world_y
        self._pitch_rad = math.radians(pitch)
        self._fixed_offset = fixed_offset
        self._ppm = pixels_per_meter
        self._crop = crop_size
        self.device = device

    def __call__(self, camera_locations: torch.Tensor, normalized: bool = True) -> torch.Tensor:
        """
        Converts 2D camera image coordinates to BEV top-down map coordinates,
        taking into account camera pitch.

        Args:
            camera_locations (torch.Tensor): (..., 2), coordinates in [-1, 1] or pixels.
            normalized (bool): If True, input is normalized to [-1, 1].

        Returns:
            torch.Tensor: (..., 2), projected BEV map coordinates in pixel space.
        """
        if normalized:
            camera_locations = (camera_locations + 1) * self._img_size / 2

        w, h = self._img_size
        cx, cy = w / 2, h / 2
        f = w / (2 * np.tan(self._fov * np.pi / 360))  # horizontal focal length

        # Compute image-based direction vector (before pitch)
        x_img = (camera_locations[..., 0] - cx) / f
        y_img = (camera_locations[..., 1] - cy) / f
        z_img = torch.ones_like(x_img)

        dirs = torch.stack([x_img, y_img, z_img], dim=-1)  # shape (..., 3)

        # Rotation around X-axis (camera pitch)
        pitch = self._pitch_rad
        cos_p, sin_p = math.cos(pitch), math.sin(pitch)
        R = torch.tensor([
            [1,     0,      0],
            [0,  cos_p, -sin_p],
            [0,  sin_p,  cos_p]
        ], device=self.device, dtype=torch.float32)

        dirs_world = dirs @ R.T  # rotate each direction vector, shape (..., 3)

        # Intersect ray with ground plane at y = 0
        dy = dirs_world[..., 1]
        dy = torch.clamp(dy, min=1e-5)  # avoid division by zero
        scale = self._world_y / dy

        world_x = dirs_world[..., 0] * scale
        world_z = dirs_world[..., 2] * scale

        # Convert to top-down pixel coordinates
        map_output = torch.stack([world_x, world_z], dim=-1)  # (..., 2) in meters
        map_output *= self._ppm  # meters to pixels

        map_output[..., 1] = self._crop - map_output[..., 1]  # flip Y
        map_output[..., 0] += self._crop / 2  # center X
        map_output[..., 1] += self._fixed_offset * self._ppm  # push Y down

        return map_output
    

class OGCoordConverter:
    def __init__(self, w=320, h=240, fov=90, world_y=2.0, fixed_offset=2.0, device='cuda', pixels_per_meter=5.0, crop_size=192):
        self._img_size = torch.FloatTensor([w,h]).to(device)
        self.ppm = pixels_per_meter
        self.crop_size = crop_size
        self._fov = fov
        self._world_y = world_y
        self._fixed_offset = fixed_offset
    
    def __call__(self, camera_locations):
        camera_locations = (camera_locations + 1) * self._img_size/2
        w, h = self._img_size
        
        cx, cy = w/2, h/2

        f = w /(2 * np.tan(self._fov * np.pi / 360))
    
        xt = (camera_locations[...,0] - cx) / f
        yt = (camera_locations[...,1] - cy) / f

        world_z = self._world_y / yt
        world_x = world_z * xt
        
        map_output = torch.stack([world_x, world_z],dim=-1)
    
        map_output *= self.ppm
        map_output[...,1] = self.crop_size - map_output[...,1]
        map_output[...,0] += self.crop_size/2
        map_output[...,1] += self._fixed_offset*self.ppm
        
        return map_output


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

        # FIXME: x and y are reversed here, do check it out if you face any problems
        # NOTE: considerably low loss when yx than xy, check BZ version
        pos_y, pos_x = np.meshgrid(
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
    """
    Computes softmax-based spatial centroids using custom coordinate mapping:
        - x in [-1, 1], left to right
        - y in [0, 1], bottom to top

    Input shape:  (N, C, H, W)
    Output shape: (N, C, 2)

    Example:
        softmax = SpatialSoftmaxBZ(height=64, width=64)
        out = softmax(torch.rand(8, 32, 64, 64))  # out.shape: (8, 32, 2)
    """
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
