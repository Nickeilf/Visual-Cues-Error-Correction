# -*- coding: utf-8 -*-
import math

import torch
import torch.nn.functional as F
from torch import nn

from ..utils.nn import get_activation_fn


class TextGating(nn.Module):
    
    def __init__(self, text_dim, feat_dim, mid_dim, dropout=0.5, att_activ='tanh', gating_type="mlp"):
        super().__init__()
        self.mid_dim = mid_dim
        self.activ = get_activation_fn(att_activ)
        self.gating_type = gating_type

        self.text2mid = nn.Linear(text_dim, mid_dim, bias=False)
        self.feat2mid = nn.Linear(feat_dim, mid_dim, bias=False)
        self.dp = nn.Dropout(p=dropout)

        if gating_type == "mlp":
            self.mlp = nn.Linear(self.mid_dim, 1, bias=False)

    # feat: 1 * B * H
    # text_hidden: S * B * H
    # return: S * B
    def forward(self, feat, text_hidden):
         # SxBxC
        text_ = self.dp(self.text2mid(text_hidden))
        # TxBxC
        feat_ = self.dp(self.feat2mid(feat))

        # text_ = text_hidden
        # feat_ = feat

        if self.gating_type == "dot":
            # S * B
            weight = torch.sigmoid(torch.bmm(feat_.permute(1, 0, 2), text_.permute(1, 2, 0)).squeeze(1).t())
        else:
            inner_sum = text_ + feat_
            weight = torch.sigmoid(self.mlp(self.activ(inner_sum)).squeeze(-1))
        return weight


