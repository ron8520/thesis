import torch.nn as nn
from SwinUnetV2.network.vision_transformer import SwinUnet


class MySwinUnetV2(nn.Module):
  def __init__(self):
    super(MySwinUnetV2, self).__init__()
    self.backbone = SwinUnet()
    self.backbone.load_from('./SwinUnet/pretrained_ckpt/swin_tiny_patch4_window7_224.pth')

  def forward(self, x, batch_positions=None):
    out = self.backbone(x, batch_positions)
    return out