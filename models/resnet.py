"""
Custom ResNet Backbone Module
-----------------------------

This module provides a customizable and flexible implementation of the ResNet architecture.
It extends PyTorch's standard ResNet by supporting arbitrary input channels (e.g., 1, 3, 7),
returning feature maps instead of classification logits, and enabling partial loading of pretrained weights.

Input Shape:
-------------
    x: Tensor of shape [B, C, H, W]
        - B: Batch size
        - C: Number of input channels (e.g., 3 for RGB, 7 for multi-modal input)
        - H, W: Image height and width (must be divisible by 32 for standard ResNet)

Output Shape:
--------------
    out: Tensor of shape [B, C_out, H_out, W_out]
        - C_out: 512 (ResNet18/34), 2048 (ResNet50+)
        - H_out = H // 32
        - W_out = W // 32

Key Features:
-------------
- Supports ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
- Handles custom input channels (e.g., RGBD, multispectral, etc.)
- Returns final feature map from last convolutional block (layer4)
- Can load pretrained ImageNet weights (only when input_channel == 3)
- Enables zero-initialization of residual branches for stable training

Source: https://github.com/dotchen/LearningByCheating/blob/release-0.9.6/bird_view/models/resnet.py
"""

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

# URLs for loading pretrained weights
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """Basic residual block used in ResNet-18 and ResNet-34."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck residual block used in ResNet-50 and deeper."""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet backbone that outputs feature maps from the last block (layer4), not classification logits.

    Input:
        x: Tensor of shape [B, C, H, W]
            - B: Batch size
            - C: input_channel (default: 3 or 7)
            - H, W: image size (should be divisible by 32)

    Output:
        Tensor of shape [B, C_out, H_out, W_out]
            - C_out: 512 (resnet18/34) or 2048 (resnet50+)
            - H_out = H // 32, W_out = W // 32
    """

    def __init__(self, block, layers, input_channel=7, num_classes=1000, zero_init_residual=False, bias_first=True):
        super(ResNet, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=bias_first)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Not used in forward()
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Identity initialization
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        """Constructs a sequence of residual blocks with optional downsampling."""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the ResNet backbone.

        Input:
            x: [B, C, H, W] — batch of images
        Output:
            out: [B, C_out, H_out, W_out] — feature maps
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


# Configurations for different ResNet variants
model_funcs = {
    'resnet18': (BasicBlock, [2, 2, 2, 2], 512),
    'resnet34': (BasicBlock, [3, 4, 6, 3], 512),
    'resnet50': (Bottleneck, [3, 4, 6, 3], 2048),
    'resnet101': (Bottleneck, [3, 4, 23, 3], 2048),
    'resnet152': (Bottleneck, [3, 8, 36, 3], 2048),
}


def get_resnet(model_name='resnet18', pretrained=False, **kwargs):
    """
    Builds a custom ResNet feature extractor.

    Args:
        model_name (str): Name of ResNet variant. One of:
            ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        pretrained (bool): Whether to load pretrained weights (only if input_channel == 3).
        **kwargs: Extra args passed to ResNet, including:
            - input_channel (int): Number of input channels (e.g., 3 for RGB, 7 for multispectral)
            - zero_init_residual (bool): If True, zero-initialize residual branches
            - bias_first (bool): Whether to add bias to the first conv layer

    Returns:
        model (nn.Module): ResNet model returning [B, C_out, H_out, W_out] feature maps.
        c_out (int): Number of output channels in the final feature map.

    Example:
        >>> model, c_out = get_resnet('resnet50', input_channel=7, pretrained=False)
        >>> x = torch.randn(4, 7, 224, 224)
        >>> y = model(x)  # y.shape = [4, 2048, 7, 7]
    """
    block, layers, c_out = model_funcs[model_name]
    model = ResNet(block, layers, **kwargs)

    if pretrained and kwargs.get('input_channel', 3) == 3:
        url = model_urls[model_name]
        print("Loading ResNet weights from:", url)
        model.load_state_dict(model_zoo.load_url(url))

    return model, c_out
