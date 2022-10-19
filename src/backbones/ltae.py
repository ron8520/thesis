import copy
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
from einops import repeat, rearrange

from src.backbones.positional_encoding import PositionalEncoder


class LTAE2d(nn.Module):
    def __init__(
        self,
        in_channels=128,
        n_head=16,
        d_k=4,
        mlp=[256, 128],
        dropout=0.2,
        d_model=256,
        T=1000,
        return_att=False,
        positional_encoding=True,
    ):
        """
        Lightweight Temporal Attention Encoder (L-TAE) for image time series.
        Attention-based sequence encoding that maps a sequence of images to a single feature map.
        A shared L-TAE is applied to all pixel positions of the image sequence.
        Args:
            in_channels (int): Number of channels of the input embeddings.
            n_head (int): Number of attention heads.
            d_k (int): Dimension of the key and query vectors.
            mlp (List[int]): Widths of the layers of the MLP that processes the concatenated outputs of the attention heads.
            dropout (float): dropout
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model.
            T (int): Period to use for the positional encoding.
            return_att (bool): If true, the module returns the attention masks along with the embeddings (default False)
            positional_encoding (bool): If False, no positional encoding is used (default True).
        """
        super(LTAE2d, self).__init__()
        self.in_channels = in_channels
        self.mlp = copy.deepcopy(mlp)
        self.return_att = return_att
        self.n_head = n_head

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)
        else:
            self.d_model = in_channels
            self.inconv = None

        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                self.d_model // n_head, T=T, repeat=n_head
            )
        else:
            self.positional_encoder = None

        self.attention_heads = MultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model
        )
        # self.in_norm = nn.GroupNorm(
        #     num_groups=n_head,
        #     num_channels=self.in_channels,
        # )
        self.in_norm = nn.LayerNorm(self.in_channels)
        # self.out_norm = None
        # self.out_norm = nn.GroupNorm(
        #     num_groups=n_head,
        #     num_channels=mlp[-1],
        # )

        # layers = []
        # for i in range(len(self.mlp) - 1):
        #     layers.extend(
        #         [
        #             nn.Linear(self.mlp[i], self.mlp[i + 1]),
        #             nn.LayerNorm(self.mlp[i + 1]),
        #             nn.GELU(),
        #         ]
        #     )

        self.proj = nn.Linear(self.mlp[0], self.mlp[1])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch_positions=None, pad_mask=None, return_comp=False):
        # sz_b, seq_len, d, h, w = x.shape
        B, L, C = x.shape
        SZ_B, T = batch_positions.shape
        if pad_mask is not None:
            # pad_mask = (
            #     pad_mask.unsqueeze(-1)
            #     .repeat((1, 1, h))
            #     .unsqueeze(-1)
            #     .repeat((1, 1, 1, w))
            # )  # BxTxHxW

            pad_mask = pad_mask.unsqueeze(-1).repeat((1, 1, L))
            pad_mask = pad_mask.permute(0, 2, 1).contiguous().view(SZ_B * L, T)

            # pad_mask = (
            #     pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            # )
        out = rearrange(x, '(b t) n c -> (b n) t c', t=T, b=SZ_B)
        out = self.in_norm(out)

        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positional_encoder is not None:
            # bp = (
            #     batch_positions.unsqueeze(-1)
            #     .repeat((1, 1, h))
            #     .unsqueeze(-1)
            #     .repeat((1, 1, 1, w))
            # )  # BxTxHxW

            # bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)

            bp = batch_positions.unsqueeze(-1).repeat((1, 1, L))
            bp = bp.permute(0, 2, 1).contiguous().view(SZ_B * L, T)
            out = out + self.positional_encoder(bp)

        out, attn = self.attention_heads(out, pad_mask=pad_mask)

        print(attn.shape)
        attn = attn.view(self.n_head, SZ_B, int(sqrt(L)), int(sqrt(L)), T).permute(
            0, 1, 4, 2, 3
        )  # head x b x t x h x w
        print(attn.shape)
        print("--------------")
        print(out.shape)
        out = (
            out.permute(1, 0, 2).contiguous().view(SZ_B * L, -1)
        )  # Concatenate heads
        out = self.dropout(self.proj(out))
        # out = self.out_norm(out) if self.out_norm is not None else out
        print(out.shape)
        out = out.view(B, L, C)
        print(out.shape)
        # out = out.view(sz_b, h, w, -1).permute(0, 3, 1, 2)

        if self.return_att:
            return out, attn
        else:
            return out


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, n_head, d_k, d_in, proj_drop=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.Q = nn.Parameter(torch.zeros((n_head, d_k))).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.scale = d_k ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(proj_drop)

    def forward(self, v, pad_mask=None):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = v.size()

        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(
            -1, d_k
        )  # (n*b) x d_k

        # q = torch.stack([self.Q for _ in range(sz_b)], dim=1)
        # q = q.unsqueeze(-1).repeat((1, 1, 1, seq_len)).permute(1, 0, 3, 2).contiguous()

        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk
        print(q.shape)
        print(k.shape)
        if pad_mask is not None:
            pad_mask = pad_mask.repeat(
                (n_head, 1)
            )  # replicate pad_mask for each head (nxb) x lk
            # pad_mask = repeat(pad_mask, 'b t -> b h t', h=n_head)

        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(
            n_head * sz_b, seq_len, -1
        )

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        attn = attn.masked_fill(pad_mask, -1e3)
        attn = self.softmax(attn)
        attn = self.dropout(attn)

        output = attn @ v
        print(output.shape)
        attn = attn.view(n_head, sz_b, 1, seq_len)
        attn = attn.squeeze(dim=2)

        output = output.view(n_head, sz_b, 1, d_in // n_head)
        output = output.squeeze(dim=2)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, pad_mask=None, return_comp=False):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e3)
        if return_comp:
            comp = attn
        # compat = attn
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn
