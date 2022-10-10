import os
import copy
import torch
import torch.nn as nn
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from .swin_unet_v2 import SwinTransformerSys
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from src.models.poolformer import Feature_aliasing, Feature_reduce


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
    self.load_from('/content/drive/MyDrive/transformer-experiment/SwinUnet/pretrained_ckpt/swin_tiny_patch4_window7_224.pth', self.s2_swin_unet)
    self.load_from('/content/drive/MyDrive/transformer-experiment/SwinUnet/pretrained_ckpt/swin_tiny_patch4_window7_224.pth', self.s1a_swin_unet)
    self.load_from('/content/drive/MyDrive/transformer-experiment/SwinUnet/pretrained_ckpt/swin_tiny_patch4_window7_224.pth', self.s1d_swin_unet)
    self.dims = [96, 192, 384, 768]
    self.reduce_dims = Feature_reduce(self.dims[-1] * 3, self.dims[-1])
    
    self.concat_dims = nn.ModuleList()
    for i in range(len(self.dims)):
      self.concat_dims.append(
        nn.Sequential(
          Feature_reduce(self.dims[i] * 3, self.dims[i]),
          Rearrange('b c h w -> b (h w) c')
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
      s2_downsamples[i] = torch.cat([element, s1a_downsamples[i], s1d_downsamples[i]], dim=1)
      s2_downsamples[i] = self.concat_dims[i](s2_downsamples[i])

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
        if "model"  not in pretrained_dict:
            print("---start load pretrained modle by splitting---")
            pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
            for k in list(pretrained_dict.keys()):
                if "output" in k:
                    print("delete key:{}".format(k))
                    del pretrained_dict[k]
            msg = model.load_state_dict(pretrained_dict,strict=False)
            # print(msg)
            return
        pretrained_dict = pretrained_dict['model']
        print("---start load pretrained modle of swin encoder---")

        model_dict = model.state_dict()
        full_dict = copy.deepcopy(pretrained_dict)
        for k, v in pretrained_dict.items():
            if "layers." in k:
                current_layer_num = 3-int(k[7:8])
                current_k = "layers_up." + str(current_layer_num) + k[8:]
                full_dict.update({current_k:v})
        for k in list(full_dict.keys()):
            if k in model_dict:
                if full_dict[k].shape != model_dict[k].shape:
                    print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                    del full_dict[k]

        msg = model.load_state_dict(full_dict, strict=False)
        # print(msg)
    else:
        print("none pretrain")
