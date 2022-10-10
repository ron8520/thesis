import torch.nn as nn
from VSASwinUnetV2.network.vision_transformer import VSASwinUnet


class MyVSASwinUnetV2(nn.Module):
  def __init__(self):
    super(MyVSASwinUnetV2, self).__init__()
    self.backbone = VSASwinUnet()
    self.backbone.load_from('/content/drive/MyDrive/transformer-experiment/SwinUnet/pretrained_ckpt/swin_tiny_patch4_window7_224.pth')

  def forward(self, x, batch_positions=None):
    out = self.backbone(x, batch_positions)
    return out