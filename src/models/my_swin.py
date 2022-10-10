import os
import copy
import torch
import torch.nn as nn
from einops import rearrange, reduce
from src.models.swin_transformer_v2 import swin_transformer_v2_t
from src.models.myformer import Channel_decrease
from src.backbones.ltae import LTAE2d
from src.backbones.utae import Temporal_Aggregator


class SwinNet(nn.Module):
  def __init__(
    self,
    checkpoint = "",
    config = ""
  ):
    super(SwinNet, self).__init__(),
    self.checkpoint = checkpoint,
    self.config = config
    self.pad_value = 0

    self.swin = swin_transformer_v2_t(
      in_channels=10,
      window_size=8,
      input_resolution=(128, 128),
      sequential_self_attention=False,
      use_checkpoint=True
    )
    self.swin.load_state_dict(torch.load(checkpoint), strict=False)

    ## Baseline temporal attention
    self.temporal_encoder = LTAE2d(
            in_channels=768,
            d_model=256,
            n_head=16,
            mlp=[256, 768],
            return_att=True,
            d_k=4,
    )
    self.temporal_aggregator = Temporal_Aggregator(mode="att_group")


    self.up4 = nn.Sequential(
      nn.Upsample(scale_factor = 2, mode="bilinear", align_corners=True),
      Channel_decrease(768, 384)
    )

    self.conv4 = Channel_decrease(768, 384)
    self.conv4_2 = Channel_decrease(384, 384)

    self.up3 = nn.Sequential(
      nn.Upsample(scale_factor = 2, mode="bilinear", align_corners=True),
      Channel_decrease(384, 192)
    )

    self.conv3 = Channel_decrease(384, 192)
    self.conv3_2 = Channel_decrease(192, 192)

    self.up2 = nn.Sequential(
      nn.Upsample(scale_factor = 2, mode="bilinear", align_corners=True),
      Channel_decrease(192, 96)
    )

    self.conv2 = Channel_decrease(192, 96)
    self.conv2_2 = Channel_decrease(96, 96)


    self.up1 = nn.Sequential(
      nn.Upsample(scale_factor = 4, mode="bilinear", align_corners=True),
      Channel_decrease(96, 96),
      Channel_decrease(96, 20)
    )
    # self.decoder = UperNet(
    #   num_classes = 20
    # )
                            
    # self.decoder = UPerHead(
    #   in_channels=[96, 192, 384, 768],
    #   in_index=[0, 1, 2, 3],
    #   pool_scales=(1, 2, 3, 6),
    #   channels=512,
    #   dropout_ratio=0.1,
    #   num_classes=19,
    #   align_corners=False)
      
    # self.assist = FCNHead(
    #   in_channels=384,
    #   in_index=2,
    #   channels=256,
    #   num_convs=1,
    #   concat_input=False,
    #   dropout_ratio=0.1,
    #   num_classes=19,
    #   align_corners=False)
    
    # 删除有关分类类别的权重
    # for k in list(self.weights_dict.keys()):
    #     if "head" in k:
    #         del self.weights_dict[k]
        # del self.weights_dict['patch_embed.proj.weight']
        # del self.weights_dict['patch_embed.proj.bias']
    # print("-----------------------------")
    # print(self.backbone.load_state_dict(self.weights_dict, strict=False))
    
    
  def forward(self, x, batch_positions=None):
    pad_mask = (
      (x == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
    )  # BxT pad mask
    B, T, C, H, W = x.shape
    x = rearrange(x, 'b t c h w -> (b t) c h w')
    features = self.swin(x)
    # out = self.decoder(out)

    for i, elements in enumerate(features):
      b, c, h, w = elements.shape
      features[i] = rearrange(elements, 
        '(b t) c h w -> b t c h w', b=B, t=T)

    out, att = self.temporal_encoder(
      features[-1], 
      batch_positions=batch_positions, 
      pad_mask=pad_mask
    )
    out = self.up4(out)
    skip = self.temporal_aggregator(
      features[-2], pad_mask=pad_mask, attn_mask=att
    )
    out = torch.cat((out, skip), dim=1)
    out = self.conv4(out)

    out = self.up3(out)
    skip = self.temporal_aggregator(
      features[1], pad_mask=pad_mask, attn_mask=att
    )
    out = torch.cat((out, skip), dim=1)
    out = self.conv3(out)

    out = self.up2(out)
    skip = self.temporal_aggregator(
      features[0], pad_mask=pad_mask, attn_mask=att
    )
    out = torch.cat((out, skip), dim=1)
    out = self.conv2(out)

    out = self.up1(out)
    # out = rearrange(out, '(b t) c h w -> b t c h w', b=B, t=T)
    # out = reduce(out, 'b t c h w -> b c h w', 'mean')

    return out





