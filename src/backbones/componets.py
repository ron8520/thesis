import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Reduce, Rearrange


class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)

class MBConv(nn.Module):
    def __init__(self, dim_in,
                 input_resolution,
                 dim_out,
                 expansion_rate=4,
                 shrinkage_rate=0.25,
                 dropout=0.):
        super().__init__()
        hidden_dim = int(expansion_rate * dim_out)
        self.input_resolution = input_resolution
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            SqueezeExcitation(hidden_dim, shrinkage_rate=shrinkage_rate),
            nn.Conv2d(hidden_dim, dim_out, 1),
            nn.BatchNorm2d(dim_out)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = rearrange(x, 'b (h w) c -> b c h w', h=self.input_resolution[0], w=self.input_resolution[1])
        out = self.net(out)
        out = self.drop(out)
        out = rearrange(out, 'b c h w -> b (h w) c')
        return x + out

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
        self.norm = GroupNorm(in_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class Feature_reduce(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = GroupNorm(out_channels)
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
