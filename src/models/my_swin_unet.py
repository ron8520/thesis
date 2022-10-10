import os
import copy
import torch
import torch.nn as nn
from einops import rearrange, reduce
from SwinUnet.networks.vision_transformer import SwinUnet
from src.backbones.utae import ConvLayer

class MySwinUnet(nn.Module):
  def __init__(
    self
  ):
    super(MySwinUnet, self).__init__()
    self.s2 = SwinUnet(in_chans=64, d_model=256, n_head=16)
    self.s1a = SwinUnet(in_chans=16, d_model=128, n_head=8)
    self.s1d = SwinUnet(in_chans=16, d_model=128, n_head=8)
    self.s2.load_from('/content/drive/MyDrive/transformer-experiment/SwinUnet/pretrained_ckpt/swin_tiny_patch4_window7_224.pth')
    self.s1a.load_from('/content/drive/MyDrive/transformer-experiment/SwinUnet/pretrained_ckpt/swin_tiny_patch4_window7_224.pth')
    self.s1d.load_from('/content/drive/MyDrive/transformer-experiment/SwinUnet/pretrained_ckpt/swin_tiny_patch4_window7_224.pth')

    self.decoder = ConvLayer(nkernels=[60, 30, 30, 20], last_relu=False)

  def forward(self, x, batch_positions=None):
    out_s2 = self.s2(x['S2'], batch_positions['S2'])
    out_s1a = self.s1a(x['S1A'], batch_positions['S1A'])
    out_s1d = self.s1d(x['S1D'], batch_positions['S1D'])
    
    out = self.decoder(torch.cat([out_s2, out_s1a, out_s1d], dim=1))

    return out