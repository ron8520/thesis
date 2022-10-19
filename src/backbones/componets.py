import torch.nn as nn


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

class Feature_aliasing(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)

class Feature_reduce(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)

class Prediction_head(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1)
        self.norm1 = GroupNorm(in_channels // 2)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels // 2, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        return self.conv2(x)