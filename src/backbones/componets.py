import numpy as np
import torch
import torch.nn as nn
from einops import repeat, rearrange


class SimplifiedScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, num_heads, dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SimplifiedScaledDotProductAttention, self).__init__()

        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.num_heads = num_heads

        self.fc_o = nn.Linear(num_heads * self.d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, N=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = queries.view(b_s, nq, self.num_heads, self.d_k).permute(0, 2, 1, 3)  # (b_s, head, nq, d_k)
        k = keys.view(b_s, nk, self.num_heads, self.d_k).permute(0, 2, 3, 1)  # (b_s, head, d_k, nk)
        v = values.view(b_s, nk, self.num_heads, self.d_v).permute(0, 2, 1, 3)  # (b_s, head, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, N)
            attention_mask = attention_mask.permute(0, 2, 1).contiguous().view(b_s, nq)
            attention_mask = repeat(attention_mask, 'b t -> b h t', h=self.num_heads)
            attention_mask = repeat(attention_mask, 'b h t -> b h t t1', t1=nq)
            att = att.masked_fill(attention_mask, -np.inf)
        att = self.softmax(att)
        att = self.dropout(att)

        out = (att @ v).permute(0, 2, 1, 3).contiguous().view(b_s, nq,
                                                                            self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


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
