import math

import torch
from easydict import EasyDict
from torch import nn
from torch.nn import functional as F

from controlledshifts.models.components.attention import AbstractAttentionLayer, KVCache, MultiHeadAttention
from controlledshifts.models.components.common import MLP, LayerNorm, Residual, RotaryPositionEmbedding


class CrossAttentionBlock(nn.Module):
    """Cross-Attention Block module."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        widening_factor: int,
        num_heads: int,
        dropout: float,
        bias: bool,
    ) -> None:
        """Cross-Attention intialization method.

        Inputs
        ------
            config[EasyDict]: dictionary with configuration parameters.
        """
        super().__init__()
        assert hidden_size % num_heads == 0

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.kv_size = 2

        # architecture
        self.ln_1 = LayerNorm(input_size, bias=bias)
        self.q = nn.Linear(input_size, hidden_size, bias=bias)
        self.kv = nn.Linear(hidden_size, self.kv_size * hidden_size, bias=bias)
        self.c_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.ln_2 = LayerNorm(hidden_size, bias=bias)
        self.mlp = MLP(num_channels=hidden_size, widening_factor=widening_factor, bias=bias)

        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.mask_val = -torch.finfo(torch.float32).max
        # TODO: enable speedup
        # self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        # # if not self.flash:
        # #     print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        # #     # causal mask to ensure that attention is only applied to the left in the input sequence
        # #     self.register_buffer(
        # #         "bias", torch.tril(torch.ones(T_size, T_size))
        # #             .view(1, 1, T_size, T_size))
        # self.register_buffer(
        #     "causal_mask",
        #     torch.tril(torch.ones(T_size, T_size)).view(1, 1, 1, T_size, T_size),
        # )
        # scene_mask = torch.zeros(T_size)
        # scene_mask[: hist_len] = 1
        # self.register_buffer("scene_mask", scene_mask.view(-1, 1, 1, 1))

    def attn(
        self,
        x: torch.Tensor,
        cx: torch.Tensor | None = None,
        mask_x: torch.Tensor | None = None,
        mask_cx: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Performs Multi-Head Cross-Attention (MHA).

        Input
        -----
            x[torch.tensor(B, A, T, D)]: input tensor used as keys (K) and values (V).
                B: batch size
                A: number of agents
                T: trajectory length
                D: embedding size
            cx[torch.tensor(B, T, D)]: input tensor used as queries (Q).

        Output
        ------
            y[torch.tensor(B, A, T, D)]: cross-attended output.
        """
        B, A, T, D = x.size()
        B, P, _ = cx.size()

        # Query shape: (batch_size, time, agents, hidden_size)
        Q = self.q(x)
        Q = Q.view(B, T, A, self.num_heads, D // self.num_heads).transpose(4, 3)

        # Key, Values shape: (batch_size, polylines_features, hidden_size)
        K, V = self.kv(cx).split(self.hidden_size, dim=-1)
        K = K.view(B, P, self.num_heads, D // self.num_heads)
        V = V.view(B, P, self.num_heads, D // self.num_heads).transpose(2, 3)

        # att: (B, P, T, A, H)
        QK = torch.einsum('bphd,btadh->bptah', K, Q)
        att = QK * (1.0 / math.sqrt(self.num_heads))

        # agent mask
        if mask_x is not None:
            # mask shape: (B, A, T) -> (B, T, A, H)
            mask = mask_x.transpose(1, 2).unsqueeze(-1).repeat(1, 1, 1, self.num_heads)
            # mask shape: (B, T, A, H) -> (B, P, T, A, H)
            mask = mask.unsqueeze(1).repeat(1, P, 1, 1, 1)
            att = att.masked_fill_(mask, self.mask_val)

        # map mask
        if mask_cx is not None:
            # mask shape: (B, P) -> (B, P, T, A, H)
            mask = mask_cx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, T, A, self.num_heads)
            att = att.masked_fill_(mask, self.mask_val)

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # re-assemble all head outputs side by side
        # y: (B, A, H, T, HD)
        y = torch.einsum('bptan,bphn->btahn', att, V)
        y = y.transpose(2, 3).contiguous().view(B, A, T, D)  # (B, A, T, D=hidden_size)

        y = self.c_proj(y)  # (A, T, D=hidden_size)
        return self.resid_dropout(y)  # (A, T, D=hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        cx: torch.Tensor | None = None,
        mask_x: torch.Tensor | None = None,
        mask_cx: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Model's forward function.

        Input
        -----
            x[torch.tensor]: input tensor to be self-attended.

        Output
        ------
            x[torch.tensor]: attended output tensor.
        """
        x = self.ln_1(x)
        x = x + self.attn(x, cx, mask_x=mask_x, mask_cx=mask_cx)
        x = self.ln_2(x)
        return x + self.mlp(x)['last_hidden_state']


class CrossAttentionLayer(AbstractAttentionLayer):
    def __init__(
        self,
        num_heads: int,
        num_q_input_channels: int,
        num_kv_input_channels: int,
        num_qk_channels: int | None = None,
        num_v_channels: int | None = None,
        max_heads_parallel: int | None = None,
        causal_attention: bool = False,
        widening_factor: int = 1,
        dropout: float = 0.0,
        residual_dropout: float = 0.0,
        attention_residual: bool = True,
        qkv_bias: bool = True,
        out_bias: bool = True,
        mlp_bias: bool = True,
    ):
        cross_attn = CrossAttention(
            num_heads=num_heads,
            num_q_input_channels=num_q_input_channels,
            num_kv_input_channels=num_kv_input_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            max_heads_parallel=max_heads_parallel,
            causal_attention=causal_attention,
            dropout=dropout,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
        )

        self.num_qk_channels = cross_attn.attention.num_qk_channels
        self.num_v_channels = cross_attn.attention.num_v_channels

        super().__init__(
            Residual(cross_attn, residual_dropout) if attention_residual else cross_attn,
            Residual(MLP(num_q_input_channels, widening_factor, bias=mlp_bias), residual_dropout),
        )


class CrossAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_q_input_channels: int,
        num_kv_input_channels: int,
        num_qk_channels: int | None = None,
        num_v_channels: int | None = None,
        max_heads_parallel: int | None = None,
        causal_attention: bool = False,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
    ):
        """Pre-layer-norm cross-attention (see `MultiHeadAttention` for attention details)."""
        super().__init__()
        self.q_norm = nn.LayerNorm(num_q_input_channels)
        self.kv_norm = nn.LayerNorm(num_kv_input_channels)
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            num_q_input_channels=num_q_input_channels,
            num_kv_input_channels=num_kv_input_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            max_heads_parallel=max_heads_parallel,
            causal_attention=causal_attention,
            dropout=dropout,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
        )

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor | None = None,
        x_kv_prefix: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
        rot_pos_emb_q: RotaryPositionEmbedding | None = None,
        rot_pos_emb_k: RotaryPositionEmbedding | None = None,
        kv_cache: KVCache | None = None,
    ):
        """Pre-layer-norm cross-attention of query input `x_q` to key/value input (`x_kv` or `x_kv_prefix`).

        If `x_kv_prefix` is defined, the entire key/value input is a concatenation of `x_kv_prefix` and `x_q` along
        the sequence dimension. In this case, the query attends to itself at the end of the key/value sequence (use
        case: Perceiver AR). If `x_kv_prefix` is not defined, `x_kv` is the entire key/value input.
        """
        x_q = self.q_norm(x_q)

        if x_kv is None:
            x_kv_prefix = self.kv_norm(x_kv_prefix)
            x_kv = torch.cat([x_kv_prefix, x_q], dim=1)
        else:
            x_kv = self.kv_norm(x_kv)

        return self.attention(
            x_q,
            x_kv,
            pad_mask=pad_mask,
            rot_pos_emb_q=rot_pos_emb_q,
            rot_pos_emb_k=rot_pos_emb_k,
            kv_cache=kv_cache,
        )
