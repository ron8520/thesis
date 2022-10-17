import copy
import torch
import torch.nn as nn
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from .swin_unet_v2 import SwinTransformerSys, Mlp_Relu, Mlp, window_partition, window_reverse
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from src.backbones.componets import Feature_aliasing, Feature_reduce


class Swin_multi(nn.Module):
    def __init__(self):
        super(Swin_multi, self).__init__(),
        self.s2_swin_unet = SwinTransformerSys(
            img_size=128,
            patch_size=4,
            in_chans=64,
            num_classes=20,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=8,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            drop_path_rate=0.1,
            ape=False,
            patch_norm=True,
            use_checkpoint=True,
            decoder=True
        )
        self.s1a_swin_unet = SwinTransformerSys(
            img_size=128,
            patch_size=4,
            in_chans=16,
            num_classes=20,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=8,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            drop_path_rate=0.1,
            ape=False,
            patch_norm=True,
            use_checkpoint=True,
            decoder=False
        )
        self.s1d_swin_unet = SwinTransformerSys(
            img_size=128,
            patch_size=4,
            in_chans=16,
            num_classes=20,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=8,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            drop_path_rate=0.1,
            ape=False,
            patch_norm=True,
            use_checkpoint=True,
            decoder=False
        )
        self.load_from('./SwinUnet/pretrained_ckpt/swin_tiny_patch4_window7_224.pth', self.s2_swin_unet)
        self.load_from('./SwinUnet/pretrained_ckpt/swin_tiny_patch4_window7_224.pth', self.s1a_swin_unet)
        self.load_from('./SwinUnet/pretrained_ckpt/swin_tiny_patch4_window7_224.pth', self.s1d_swin_unet)
        self.dims = [96, 192, 384, 768]
        self.reduce_dims = Feature_reduce(self.dims[-1] * 3, self.dims[-1])

        self.concat_dims = nn.ModuleList()
        for i in range(len(self.dims)):
            self.concat_dims.append(
                MultiSwinTransformerBlock(
                    dim=self.dims[i],
                    num_heads=16,
                    window_size=4,
                    mlp_ratio=4.,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=0.,
                    attn_drop=0.,
                    drop_path=0.1,
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm
                )
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.normal_(m.bias.data)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, mean=0, std=1)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            try:
                nn.init.normal_(m.bias.data)
            except AttributeError:
                pass

    def forward(self, x, batch_positions=None):
        s2_out, s2_downsamples = self.s2_swin_unet.forward_features(x['S2'], batch_positions['S2'])
        s1a_out, s1a_downsamples = self.s1a_swin_unet.forward_features(x['S1A'], batch_positions['S1A'])
        s1d_out, s1d_downsamples = self.s1d_swin_unet.forward_features(x['S1D'], batch_positions['S1D'])

        x = torch.cat([s2_out, s1a_out, s1d_out], dim=1)
        x = self.reduce_dims(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        for i, element in enumerate(s2_downsamples):
            s2_downsamples[i] = self.concat_dims[i](s2_downsamples[i], s1a_downsamples[i], s1d_downsamples[i])

        out = self.s2_swin_unet.forward_up_features(x, s2_downsamples)
        out = self.s2_swin_unet.se(out)
        out = self.s2_swin_unet.up_x4(out)

        return out

    def load_from(self, config, model):
        pretrained_path = config
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = model.load_state_dict(pretrained_dict, strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = model.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            msg = model.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")


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
        B, C, H, W = x.shape
        Ba, Ca, Ha, Wa = s1a.shape
        Bd, Cd, Hd, Wd = s1d.shape

        shortcut = rearrange(x, 'b c h w -> b (h w) c', b=B, c=C, h=H, w=W)

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
        print(x.shape)
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
            nn.Conv2d(2 * head_dim, head_dim, kernel_size=3, stride=1, padding=1),
            Rearrange('b c h w -> b (h w) c'),
            nn.LayerNorm(head_dim),
            nn.GELU()
        )

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

        # x_2a = (attn_2a @ va).transpose(1, 2).reshape(B_, N, C)
        # x_2a = (attn_2a @ va).transpose(1, 2)
        x_2a = (attn_2a @ va)

        # Sentinel-2 and Sentinel-1 cross attention (q, kd, vd)
        attn_2d = torch.einsum("bhqd, bhkd -> bhqk", q, kd) / torch.maximum(
            torch.norm(q, dim=-1, keepdim=True) * torch.norm(k, dim=-1, keepdim=True).transpose(-2, -1),
            torch.tensor(1e-06, device=q.device, dtype=q.dtype))
        attn_2d = attn_2d / torch.clip(self.tau[:, :N, :N].unsqueeze(0), min=0.01)

        attn_2d = attn_2d + relative_position_bias.unsqueeze(0)
        attn_2d = self.softmax(attn_2d)
        attn_2d = self.attn_drop(attn_2d)
        x_2d = (attn_2d @ vd)
        # x_2d = (attn_2d @ vd).transpose(1, 2).reshape(B_, N, C)

        x = rearrange(torch.cat([x_2a, x_2d], dim=3), 'b h w c -> b c h w')
        x = self.conv(x)
        print(x.shape)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops
