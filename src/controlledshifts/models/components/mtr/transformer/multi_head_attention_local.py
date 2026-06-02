# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi
# All Rights Reserved
#
# Pure-PyTorch re-implementation of MultiheadAttentionLocal.
# The original implementation relies on custom CUDA ops
# (unitraj/models/mtr/ops/attention) for sparse index-based attention.
# This version replicates the same semantics using standard PyTorch gather
# operations so that no CUDA compilation is required.

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.init import constant_
from torch.nn.init import xavier_uniform_
from torch.nn.parameter import Parameter


class MultiheadAttentionLocal(nn.Module):
    """Index-based sparse multi-head attention (pure PyTorch implementation).

    For each query token, attention is computed only over a fixed set of K
    key/value tokens identified by ``index_pair``.  Pairs marked with index
    -1 are masked out.

    Interface is kept identical to the CUDA-based original so that
    TransformerEncoderLayer / TransformerDecoderLayer can use this drop-in.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, without_weight=False, vdim=None, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(self.vdim, self.vdim, bias=True)

        self.without_weight = without_weight
        if self.without_weight:
            self.in_proj_weight = self.in_proj_bias = None
            constant_(self.out_proj.bias, 0.0)
        else:
            self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)

    def _proj(self, t, start, end):
        w = self.in_proj_weight[start:end, :]
        b = self.in_proj_bias[start:end] if self.in_proj_bias is not None else None
        return F.linear(t, w, b)

    def forward(
        self,
        query,            # (N_q, embed_dim)
        key,              # (N_k, embed_dim)
        value,            # (N_k, vdim)
        index_pair,       # (N_q, K) – indices into key/value; -1 = ignore
        query_batch_cnt,  # (B,) number of queries per batch item  [unused here]
        key_batch_cnt,    # (B,) number of keys per batch item      [unused here]
        index_pair_batch, # (N_q,) batch index of each query        [unused here]
        attn_mask=None,   # (N_q, K) additive / bool mask
        vdim=None,
        **kwargs,
    ):
        N_q, embed_dim = query.shape
        K = index_pair.shape[1]

        if vdim is None:
            vdim = self.vdim
        v_head_dim = vdim // self.num_heads
        scaling = float(self.head_dim) ** -0.5

        # Project Q, K, V
        if not self.without_weight:
            q = self._proj(query, 0, embed_dim) * scaling          # (N_q, C)
            k = self._proj(key, embed_dim, embed_dim * 2)          # (N_k, C)
            v = self._proj(value, embed_dim * 2, embed_dim * 3)    # (N_k, vdim)
        else:
            q = query * scaling
            k, v = key, value

        # Reshape for multi-head
        q = q.view(N_q, self.num_heads, self.head_dim)             # (N_q, H, D)
        k = k.view(-1, self.num_heads, self.head_dim)              # (N_k, H, D)
        v = v.view(-1, self.num_heads, v_head_dim)                 # (N_k, H, Dv)

        # Gather the K nearest neighbours for each query
        valid_mask = index_pair >= 0                               # (N_q, K)
        safe_idx = index_pair.clamp(min=0)                         # (N_q, K)

        # k_gathered: (N_q, K, H, D)
        k_gathered = k[safe_idx.view(-1)].view(N_q, K, self.num_heads, self.head_dim)
        # v_gathered: (N_q, K, H, Dv)
        v_gathered = v[safe_idx.view(-1)].view(N_q, K, self.num_heads, v_head_dim)

        # Attention weights: dot product over head dim
        # q: (N_q, H, 1, D) x k: (N_q, H, K, D) -> (N_q, H, K)
        q_exp = q.unsqueeze(2)                                     # (N_q, H, 1, D)
        k_exp = k_gathered.permute(0, 2, 1, 3)                    # (N_q, H, K, D)
        attn_weights = (q_exp * k_exp).sum(dim=-1)                 # (N_q, H, K)

        # Mask invalid pairs
        inv_mask = (~valid_mask).unsqueeze(1)                      # (N_q, 1, K)
        attn_weights = attn_weights.masked_fill(inv_mask, float("-inf"))

        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask.bool().unsqueeze(1), float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)             # (N_q, H, K)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Weighted sum of values
        v_exp = v_gathered.permute(0, 2, 1, 3)                    # (N_q, H, K, Dv)
        attn_output = (attn_weights.unsqueeze(-1) * v_exp).sum(dim=2)  # (N_q, H, Dv)
        attn_output = attn_output.reshape(N_q, vdim)               # (N_q, vdim)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        # Return weights compatible with the CUDA version: (N_q, K)
        attn_output_weights = attn_weights.permute(0, 2, 1)        # (N_q, K, H)
        return attn_output, attn_output_weights.sum(dim=-1) / self.num_heads
