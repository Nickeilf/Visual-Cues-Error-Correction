# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Scaled Dot-product attention from `Attention is all you need`.

    Arguments:

    Input:

    Output:
    """

    def __init__(self, model_dim, n_heads, dropout=0.1, with_image=False):
        super().__init__()
        self.model_dim = model_dim
        self.n_heads = n_heads
        self.with_image = with_image

        assert model_dim % n_heads == 0
        self.k_dim = self.model_dim // self.n_heads

        # Efficient linear projections for all heads
        self.lin_k = nn.Linear(
            self.model_dim, self.model_dim, bias=False)
        self.lin_q = nn.Linear(
            self.model_dim, self.model_dim, bias=False)
        self.lin_v = nn.Linear(
            self.model_dim, self.model_dim, bias=False)

        # Final output layer is independent of number of heads
        self.lin_o = nn.Linear(
            self.model_dim, self.model_dim, bias=False)

        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


    def forward(self, inputs):
        """Scaled dot-product attention forward-pass

        :param inputs: dictionary with query, key, value and mask tensors
            the shape of the tensors are (tstep, bsize, dim) except for the
            mask which is (tstep, bsize)

        :return: foo
        """
        # SxBxC
        q, k, v = inputs
        seq_len, batch_size, _ = q.shape
        
        

        qp = self.lin_q(q)

        head_view = (batch_size, -1, self.n_heads, self.k_dim)
        # qp: (bsize, head, tstep, k_dim)
        # vp: (bsize, head, tstep, k_dim)
        # kp: (bsize, head, tstep, k_dim)
        qp = self.lin_q(q).view(*head_view).transpose(1, 2)
        kp = self.lin_k(k).view(*head_view).transpose(1, 2)
        vp = self.lin_v(v).view(*head_view).transpose(1, 2)
        

        x, attn = self.attention(qp, kp, vp, mask=None, dropout=self.dropout)

        # SxBxC
        x = x.transpose(1, 2).contiguous() \
             .view(-1, batch_size, self.model_dim)
        
        x = self.lin_o(x)

        if self.with_image:
            return x[1:, :, :]
        else:
            return x