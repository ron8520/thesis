import os
import copy
import torch
import torch.nn as nn
from einops import rearrange, reduce
from src.models.poolformer import poolformer_s12, GroupNorm, Pooling
from src.models.metaformer import Attention
from src.backbones.utae import Temporal_Aggregator
from src.backbones.ltae import LTAE2d

# class PUPHead(nn.Module):
#   def __init__(
#     self,
#     d_in = d_in,
#     d_out = d_out,
    
#   ):
#     super(PUPHead, self).__init__()
  
#     # self.decoder_widths = [64, 128, 320, 512]
#     # self.out_conv = nn.Conv2d(64, 20, 3, padding=1)

#     self.up_stages = nn.Sequential(
#         nn.Conv2d(
#           d_in,
#           d_out,
#           kernel_size=4,
#           stride=2,
#           padding=1
#           ),
#           nn.BatchNorm2d(d_out),
#           nn.ReLU(),
#           nn.Upsample(scale_factor=4, mode="bilinera", align_corners=True)
#       )
#     self.n_stages = 4
#     self.decoder_widths = [64, 128, 320, 512]
#     self.decoder = nn.ModuleList(
#         UpConvBlock(
#             d_in=self.decoder_widths[i], # decoder_widths=[32, 32, 64, 128],
#             d_out=self.decoder_widths[i - 1],
#             d_skip=self.decoder_widths[i - 1],
#             k=4,
#             s=2,
#             p=1,
#             norm="batch",
#             padding_mode="reflect",
#         )
#         for i in range(self.n_stages - 2, 0, -1)
#         )
  
#   def forward(self, x, skip):
#     out = self.up(x)

#     return out


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = poolformer_s12(
          fork_feat=True,
          pretrained=True
        )
        self.decoder_dim = 128
        self.dims = [64, 128, 320, 512]
        self.num_classes = 20
        self.temporal_encoder = LTAE2d(
            in_channels=self.dims[-1],
            d_model=256,
            n_head=16,
            mlp=[256, self.dims[-1]],
            return_att=True,
            d_k=4,
        )
        self.pad_value = 0
        self.temporal_aggregator = Temporal_Aggregator(mode="att_group")
        
        # self.token_mixer_temporal = nn.ModuleList([nn.Sequential(
        #     Pooling(pool_size=3)
        # ) for i, dim in enumerate(self.dims)])
        
        # self.to_fused = nn.ModuleList([nn.Sequential(
        #     nn.Conv2d(dim, self.decoder_dim, 1),
        #     nn.Upsample(scale_factor = 2 ** (i + 2))
        # ) for i, dim in enumerate(self.dims)])

        # self.to_fused = nn.ModuleList([nn.Sequential(

        #     nn.Conv2d(dim, self.decoder_dim, 1),
        #     nn.Upsample(scale_factor = 2 ** (i + 2))
        # ) for i, dim in enumerate(self.dims)])

        self.up4 = nn.Sequential(
          nn.Upsample(scale_factor = 2, mode="bilinear", align_corners=True),
          Channel_decrease(512, 320)
        )

        self.conv4 = Channel_decrease(640, 320)

        self.up3 = nn.Sequential(
          nn.Upsample(scale_factor = 2, mode="bilinear", align_corners=True),
          Channel_decrease(320, 128)
        )

        self.conv3 = Channel_decrease(256, 128)

        self.up2 = nn.Sequential(
          nn.Upsample(scale_factor = 2, mode="bilinear", align_corners=True),
          Channel_decrease(128, 64)
        )

        self.conv2 = Channel_decrease(128, 64)


        self.up1 = nn.Sequential(
          nn.Upsample(scale_factor = 4, mode="bilinear", align_corners=True),
          Channel_decrease(64, 64),
          Channel_decrease(64, 20)
        )
        
        # self.to_segmentation = nn.Sequential(
        #     nn.Conv2d(4 * self.decoder_dim, self.decoder_dim, 1),
        #     nn.Conv2d(self.decoder_dim, self.num_classes, 1),
        # )
        
        # self.backbone = poolformer_s12(pretrained=True)
        # self.out_indices = [0, 2, 4, 6]
        # embed_dims = [64, 128, 320, 512]

        # for i_emb, i_layer in enumerate(self.out_indices):
        #     if i_emb == 0 and os.environ.get('FORK_LAST3', None):
        #         # TODO: more elegant way
        #         """For RetinaNet, `start_level=1`. The first norm layer will not used.
        #         cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
        #         """
        #         layer = nn.Identity()
        #     else:
        #         layer = nn.Identity()
        #     layer_name = f'norm{i_layer}'
        #     self.backbone.add_module(layer_name, layer)

    def forward(self, x, batch_positions=None):
        pad_mask = (
            (x == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
        )  # BxT pad mask
        B, T, C, H, W = x.shape
        out = rearrange(x, 'b t c h w -> (b t) c h w')
        feature_maps = self.backbone(out)
        
        for i, elements in enumerate(feature_maps):
          b, c, h, w = elements.shape
          feature_maps[i] = rearrange(elements, 
            '(b t) c h w -> b t c h w', b=B, t=T)

        out, att = self.temporal_encoder(
            feature_maps[-1], 
            batch_positions=batch_positions, 
            pad_mask=pad_mask
        )
        # new_feature_maps = []
        # for maps, pooling in zip(feature_maps, self.token_mixer_temporal):
        #     B_, C_, H_, W_ = maps.shape
        #     xt = maps
        #     xt = rearrange(xt, '(b t) c h w -> (b h w) t c', b=B, t=T)
        #     xt = pooling(xt)
        #     xt = rearrange(xt, '(b h w) t c -> (b t) c h w', b=B, h=H_, w=W_)
        #     out = maps + xt
        #     new_feature_maps.append(out)
        
                
        # decoder
        out = self.up4(out)
        skip = self.temporal_aggregator(
          feature_maps[-2], pad_mask=pad_mask, attn_mask=att
        )
        out = torch.cat((out, skip), dim=1)
        out = self.conv4(out)

        out = self.up3(out)
        skip = self.temporal_aggregator(
          feature_maps[1], pad_mask=pad_mask, attn_mask=att
        )
        out = torch.cat((out, skip), dim=1)
        out = self.conv3(out)

        out = self.up2(out)
        skip = self.temporal_aggregator(
          feature_maps[0], pad_mask=pad_mask, attn_mask=att
        )
        out = torch.cat((out, skip), dim=1)
        out = self.conv2(out)

        out = self.up1(out)
        # out = rearrange(out, '(b t) c h w -> b t c h w', b=B, t=T)
        # out = reduce(out, 'b t c h w -> b c h w', 'mean')
        
        return out

class Channel_decrease(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm = GroupNorm(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


