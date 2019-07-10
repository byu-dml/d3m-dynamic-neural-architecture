import math

import torch
from torch import nn

from .conv_1d import Conv1D


class Attention(nn.Module):
    def __init__(self, embed_dim, n_ctx, cfg, scale=False):
        super(Attention, self).__init__()
        n_state = embed_dim  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % cfg['n_head'] == 0
        self.register_buffer('b', torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = cfg['n_head']
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, 1, embed_dim)
        self.c_proj = Conv1D(n_state, 1, embed_dim)
        self.attn_dropout = nn.Dropout(cfg['attn_pdrop'])
        self.resid_dropout = nn.Dropout(cfg['resid_pdrop'])

    def _attn(self, query, key, value):
        w = torch.matmul(query, key)
        if self.scale:
            w = w / math.sqrt(value.size(-1))
        w = w * self.b + -1e9 * (1 - self.b)  # TF implementation method: mask_attn_weights
        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        return torch.matmul(w, value)

    def merge_heads(self, heads):
        heads = heads.permute(0, 2, 1, 3).contiguous()
        new_x_shape = heads.size()[:-2] + (heads.size(-2) * heads.size(-1),)
        return heads.view(*new_x_shape)  # in Tensorflow implementation: fct merge_states

    def split_heads(self, head, is_key=False):
        new_head_shape = head.size()[:-1] + (self.n_head, head.size(-1) // self.n_head)
        head = head.view(*new_head_shape)  # in Tensorflow implementation: fct split_states
        if is_key:
            return head.permute(0, 2, 3, 1)
        else:
            return head.permute(0, 2, 1, 3)

    def forward(self, x):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, is_key=True)
        value = self.split_heads(value)
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        return a
