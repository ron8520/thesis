import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp_Relu(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, dropout):
        super(Mlp_Relu, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
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


class MultiSwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = MultiWindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, s1a, s1d):
        B, N, C = x.shape
        H, W = int(math.sqrt(N)), int(math.sqrt(N))
        Ba, Na, Ca = s1a.shape
        Bd, Nd, Cd = s1d.shape

        shortcut = x
        x = x.view(B, H, W, C)
        s1a = s1a.view(Ba, H, W, Ca)
        s1d = s1d.view(Bd, H, W, Cd)

        shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        s1a_windows = window_partition(s1a, self.window_size)
        s1a_windows = s1a_windows.view(-1, self.window_size * self.window_size, Ca)

        s1d_windows = window_partition(s1d, self.window_size)
        s1d_windows = s1d_windows.view(-1, self.window_size * self.window_size, Cd)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, s1a_windows, s1d_windows)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        x = shifted_x
        x = x.view(B, H * W, C)

        # Swin v2
        x = self.norm1(x)

        # FFN
        x = shortcut + self.drop_path(x)

        # Swin v2
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x


class MultiWindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2

        log_relative_position_index = torch.sign(relative_coords) * torch.log(1. + relative_coords.abs())
        self.register_buffer("log_relative_position_index", log_relative_position_index)

        # Swin v2, small meta network, Eq.(3)
        self.cpb = Mlp_Relu(in_features=2,  # delta x, delta y
                            hidden_features=256,  # hidden dims
                            out_features=self.num_heads,
                            dropout=0.0)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_a = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_d = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

        self.tau = nn.Parameter(torch.ones((num_heads, window_size[0] * window_size[1],
                                            window_size[0] * window_size[1])))
        self.conv = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Dropout(proj_drop)
        )
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def get_continuous_relative_position_bias(self, N):
        # The continuous position bias approach adopts a small meta network on the relative coordinates
        continuous_relative_position_bias = self.cpb(self.log_relative_position_index[:N, :N])
        return continuous_relative_position_bias

    def forward(self, x, s1a, s1d):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        Ba_, Na, Ca = s1a.shape
        Bd_, Nd, Cd = s1d.shape

        conv_branch = rearrange(x, "b (h w) c -> b c h w", h=int(math.sqrt(N)), w=int(math.sqrt(N)))
        conv_out = self.dwconv(conv_branch)
        conv_out = rearrange(conv_out, 'b c h w -> b (h w) c')

        # Sentinel-2
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # Sentinel-1a
        qkv_a = self.qkv_a(s1a).reshape(Ba_, Na, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qa, ka, va = qkv_a[0], qkv_a[1], qkv_a[2]

        # Sentinel-1d
        qkv_d = self.qkv_d(s1d).reshape(Bd_, Nd, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qd, kd, vd = qkv_d[0], qkv_d[1], qkv_d[2]

        # Sentinel-2 and Sentinel-1a cross attention (q, ka, va)
        q = q * self.scale
        attn_2a = torch.einsum("bhqd, bhkd -> bhqk", q, ka) / torch.maximum(
            torch.norm(q, dim=-1, keepdim=True) * torch.norm(k, dim=-1, keepdim=True).transpose(-2, -1),
            torch.tensor(1e-06, device=q.device, dtype=q.dtype))
        attn_2a = attn_2a / torch.clip(self.tau[:, :N, :N].unsqueeze(0), min=0.01)

        # Swin v2
        relative_position_bias = self.get_continuous_relative_position_bias(N)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn_2a = attn_2a + relative_position_bias.unsqueeze(0)

        attn_2a = self.softmax(attn_2a)
        attn_2a = self.attn_drop(attn_2a)

        x_2a = (attn_2a @ (va + conv_out)).transpose(1, 2).reshape(B_, N, C)

        # Sentinel-2 and Sentinel-1 cross attention (q, kd, vd)
        attn_2d = torch.einsum("bhqd, bhkd -> bhqk", q, kd) / torch.maximum(
            torch.norm(q, dim=-1, keepdim=True) * torch.norm(k, dim=-1, keepdim=True).transpose(-2, -1),
            torch.tensor(1e-06, device=q.device, dtype=q.dtype))
        attn_2d = attn_2d / torch.clip(self.tau[:, :N, :N].unsqueeze(0), min=0.01)

        attn_2d = attn_2d + relative_position_bias.unsqueeze(0)
        attn_2d = self.softmax(attn_2d)
        attn_2d = self.attn_drop(attn_2d)
        x_2d = (attn_2d @ (vd + conv_out)).transpose(1, 2).reshape(B_, N, C)

        x = rearrange(torch.cat([x_2a, x_2d], dim=2), 'b (h w) c -> b c h w',
                      h=int(math.sqrt(N)), w=int(math.sqrt(N)))
        x = self.conv(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        return x