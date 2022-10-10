import torch
import torch.nn as nn
from src.backbones.utae import ConvLayer
from src.backbones.utae import UTAE

class LateUTAE(nn.Module):
    def __init__(self, light=False, aux=False, num_classes=20):
        super(LateUTAE, self).__init__()
        self.aux = aux
        self.num_classes = num_classes
        self.utae_s2 = UTAE(input_dim=10, encoder_widths=[64, 64, 64, 128], out_conv=[32, 32], k=4, s=2,
                            p=1, input_size=128, n_head=16, n_blocks=16, d_model=256, n_groups=8, d_k=8,
                            decoder_widths=[32, 32, 64, 128])

        self.utae_s1a = UTAE(input_dim=3, encoder_widths=[16, 32, 32, 64], out_conv=[16, 16], k=4, s=2,
                             p=1, input_size=128, n_head=8, n_blocks=8, d_model=128, n_groups=8, d_k=8,
                             decoder_widths=[16, 32, 32, 64])

        self.utae_s1d = UTAE(input_dim=3, encoder_widths=[16, 32, 32, 64], out_conv=[16, 16], k=4, s=2,
                             p=1, input_size=128, n_head=8, n_blocks=8, d_model=128, n_groups=8, d_k=8,
                             decoder_widths=[16, 32, 32, 64])

        self.decoder = ConvLayer(nkernels=[64, 32, 32, num_classes], last_relu=False)

        if aux:
            self.dec_s2 = ConvLayer(nkernels=[32, 32, 32, num_classes], last_relu=False)
            self.dec_s1a = ConvLayer(nkernels=[16, 16, 16, num_classes], last_relu=False)
            self.dec_s1d = ConvLayer(nkernels=[16, 16, 16, num_classes], last_relu=False)

    def forward(self, input, batch_positions=None, return_att=False, return_all=False):

        out_s2 = self.utae_s2(input['S2'], batch_positions=batch_positions['S2'])
        out_s1a = self.utae_s1a(input['S1A'], batch_positions=batch_positions['S1A'])
        out_s1d = self.utae_s1d(input['S1D'], batch_positions=batch_positions['S1D'])

        out = self.decoder(torch.cat([out_s2, out_s1a, out_s1d], dim=1))

        if self.aux:
            out_s2 = self.dec_s2(out_s2)
            out_s1a = self.dec_s1a(out_s1a)
            out_s1d = self.dec_s1d(out_s1d)

        if return_all:
            return out, (out_s2, out_s1a, out_s1d)
        else:
            return out
