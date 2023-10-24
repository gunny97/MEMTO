import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os


class Attention(nn.Module):
    def __init__(self, window_size, mask_flag=False, scale=None, dropout=0.0):
        super(Attention, self).__init__()
        self.window_size = window_size
        self.mask_flag = mask_flag
        self.scale = scale
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        '''
        queries : N x L x Head x d
        keys : N x L(s) x Head x d
        values : N x L x Head x d
        '''
        N, L, Head, C = queries.shape

        scale = self.scale if self.scale is not None else 1. / sqrt(C)

        attn_scores = torch.einsum('nlhd,nshd->nhls', queries, keys)    # N x Head x L x L
        attn_weights = self.dropout(torch.softmax(scale * attn_scores, dim=-1))

        updated_values = torch.einsum('nhls,nshd->nlhd', attn_weights, values)  # N x L x Head x d

        return updated_values.contiguous()
    

class AttentionLayer(nn.Module):
    def __init__(self, window_size, d_model, n_heads, d_keys=None, d_values=None, mask_flag=False, 
                 scale=None, dropout=0.0):
        super(AttentionLayer, self).__init__()

        self.d_keys = d_keys if d_keys is not None else (d_model // n_heads)
        self.d_values = d_values if d_values is not None else (d_model // n_heads)
        self.n_heads = n_heads
        self.d_model = d_model  # d_model = C

        # Linear projections to Q, K, V
        self.W_Q = nn.Linear(self.d_model, self.n_heads * self.d_keys)
        self.W_K = nn.Linear(self.d_model, self.n_heads * self.d_keys)
        self.W_V = nn.Linear(self.d_model, self.n_heads * self.d_values)

        self.out_proj = nn.Linear(self.n_heads * self.d_values, self.d_model)

        self.attn = Attention(window_size=window_size, mask_flag=mask_flag, scale=scale, dropout=dropout)

    def forward(self, input):
        '''
        input : N x L x C(=d_model)
        '''
        N, L, _ = input.shape

        Q = self.W_Q(input).contiguous().view(N, L, self.n_heads, -1)
        K = self.W_K(input).contiguous().view(N, L, self.n_heads, -1)
        V = self.W_V(input).contiguous().view(N, L, self.n_heads, -1)

        updated_V = self.attn(Q, K, V)  # N x L x Head x d_values
        out = updated_V.view(N, L, -1)

        return self.out_proj(out)   # N x L x C(=d_model)

