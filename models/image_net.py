"""
Input:
------
- `image`: Tensor of shape `[B, 3, H, W]` — RGB input images
- `velocity`: Tensor of shape `[B]` — scalar velocity for each sample
- `command`: Tensor of shape `[B]` — integer command index ∈ [0, ..., 6]

Output:
-------
- `location_pred`: Tensor of shape `[B, STEPS, 2]` — predicted spatial (x, y) locations
- (Optional) `location_preds`: Tensor of shape `[B, COMMANDS, STEPS, 2]` — all branch predictions
"""

from .resnet_utils import ResnetBase, NormalizeV2, SpatialSoftmax, select_branch
import torchvision.transforms.functional as tgm
import torch
import torch.nn as nn


class ImagePolicyModel(ResnetBase):
    """
    A command-conditioned visuomotor policy network based on a ResNet visual encoder.

    Args:
        backbone (str): One of {'resnet18', 'resnet34', 'resnet50'}
        warp (bool): Whether to enable warped image input (currently disabled in forward)
        pretrained (bool): If True, loads pretrained ResNet weights (ImageNet)
        all_branch (bool): If True, outputs predictions for all command branches
        **kwargs: Additional arguments passed to ResnetBase (e.g., input_channel, bias_first)

    Attributes:
        rgb_transform (nn.Module): Normalization transform (currently ImageNet mean/std)
        deconv (nn.Module): Upsampling layers for spatial prediction
        location_pred (nn.ModuleList): List of command-specific location predictors
    """

    def __init__(self, backbone, warp=False, pretrained=False, all_branch=False, steps=5, commands=4, **kwargs):
        super().__init__(backbone, pretrained=pretrained, input_channel=3, bias_first=False)

        self.c = {
            'resnet18': 512,
            'resnet34': 512,
            'resnet50': 2048
        }[backbone]

        self.warp = warp
        self.rgb_transform = NormalizeV2(
            mean=[0.485, 0.456, 0.406],  # TODO: Replace with dataset-specific statistics
            std=[0.229, 0.224, 0.225]
        )

        # Deconvolution (upsampling) head
        self.deconv = nn.Sequential(
            nn.BatchNorm2d(self.c + 128),
            nn.ConvTranspose2d(self.c + 128, 256, 3, 2, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.ReLU(True),
        )

        # Per-command prediction heads
        self.location_pred = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(64),
                nn.Conv2d(64, steps, 1, 1, 0),
                SpatialSoftmax(48, 48, steps),
            ) for _ in range(commands)
        ])

        self.all_branch = all_branch

    def forward(self, image, velocity, command):
        """
        Forward pass of the ImagePolicyModel.

        Args:
            image (Tensor): [B, 3, H, W] — input RGB images
            velocity (Tensor): [B] — scalar speed values
            command (Tensor): [B] — integer command index (0 to 6)

        Returns:
            location_pred (Tensor): [B, STEPS, 2] — (x, y) waypoints for the selected command
            location_preds (Tensor, optional): [B, COMMANDS, STEPS, 2] — all command predictions (if all_branch=True)
        """

        if self.warp:
            warped_image = tgm.warp_perspective(image, self.M, dsize=(192, 192))
            resized_image = tgm.resize(image, (192, 192))
            image = torch.cat([warped_image, resized_image], 1)
        else:
            image = tgm.resize(image, (192, 192))
        
        image = self.rgb_transform(image)           # Normalize input
        h = self.conv(image)                        # Extract features from ResNet
        b, c, kh, kw = h.size()

        if velocity.dim() == 1:
            velocity = velocity[:, None]
        velocity = velocity[..., None, None].repeat(1, 128, kh, kw)           # [B, 128, H, W]
        h = torch.cat((h, velocity), dim=1)                                   # Late fusion with image features

        h = self.deconv(h)     

        location_preds = [branch(h) for branch in self.location_pred]         # List of [B, STEPS, 2]
        location_preds = torch.stack(location_preds, dim=1)                   # [B, COMMANDS, STEPS, 2]

        location_pred = select_branch(location_preds, command)                # [B, STEPS, 2]

        if self.all_branch:
            return location_pred, location_preds

        return location_pred