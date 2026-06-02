import math

import torch
from easydict import EasyDict
from torch import nn
from torch.nn import functional as F

from controlledshifts.models.components.attention import AbstractAttentionLayer, KVCache, MultiHeadAttention
from controlledshifts.models.components.common import MLP, LayerNorm, ModuleOutput, Residual, RotaryPositionEmbedding


class FactorizedSelfAttentionBlock(nn.Module):
    """Self-Attention Block module."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        widening_factor: int,
        num_heads: int,
        dropout: float,
        bias: bool,
        across: str = "time"
    ) -> None:
        """Self-Attention intialization method.

        Inputs
        ------
            config[EasyDict]: dictionary with configuration parameters.
            across[str]: can be either 'time' or 'agents' and is used to specify whether the attention
                operations are performed across the time or agents dimentions.
        """
        super().__init__()
        assert hidden_size % num_heads == 0
        assert across in ["time", "agents"], f"Across: {across} not in {['time', 'agents']}"

        self.across = across
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout

        # architecture
        self.ln_1 = LayerNorm(input_size, bias=bias)
        self.qkv = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.c_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.ln_2 = LayerNorm(hidden_size, bias=bias)
        self.mlp = MLP(num_channels=hidden_size, widening_factor=widening_factor, bias=bias)

        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.mask_val = -torch.finfo(torch.float32).max

        # TODO: enable speedup
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        # self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
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
        """Performs Multi-Head Attention (MHA).

        Input
        -----
            x[torch.Tensor(B, A, T, D)]: input tensor over which to compute MHA.
                B: batch size
                A: number of agents
                T: trajectory length
                C: embedding size

        Output
        ------
            y[torch.Tensor(B, A, T, D)]: attended output.
        """
        B, A, T, C = x.size()

        # Query, key, values for all heads. Shapes: (B, T, A, hidden_size)
        Q, K, V = self.qkv(x).split(self.hidden_size, dim=3)

        if self.across == "time":
            # Query, key, values shapes: (batch_size, time, heads, ageents, hidden_size / heads)
            K = K.view(B, A, T, self.num_heads, C // self.num_heads).transpose(2, 3) # (B, A, H, T, HD)
            Q = Q.view(B, A, T, self.num_heads, C // self.num_heads).transpose(2, 3) # (B, A, H, T, HD)
            V = V.view(B, A, T, self.num_heads, C // self.num_heads).transpose(2, 3) # (B, A, H, T, HD)
            att = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(K.size(-1)))          # (B, A, H, T, T)

            # Apply masks
            # mask shape: (batch_size, time, agents) -> (batch_size, time, heads, agents, agents)
            if mask_x is not None:
                mask = mask_x.unsqueeze(2).repeat(1, 1, self.num_heads, 1)
                mask = mask.unsqueeze(-1).repeat(1, 1, 1, 1, T)
                att = att.masked_fill_(mask, self.mask_val)
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)

            # y shape: (batch_size, time, heads, agents, hidden_size / num_heads)
            y = att @ V
            # Re-assemble all head outputs side by side
            # y shape: (batch_size, time, agents, hidden_size)
            y = y.transpose(2, 3).contiguous().view(B, A, T, C) # (B, A, T, D=embed_size)

        elif self.across == "agents":
            K = K.view(B, A, T, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1, 4) # (T, H, A, HD)
            Q = Q.view(B, A, T, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1, 4) # (T, H, A, HD)
            V = V.view(B, A, T, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1, 4) # (T, H, A, HD)

            att = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(K.size(-1))) # (T, H, A, N)

            if mask_x is not None:
                # mask shape: (batch_size, time, agents) -> (batch_size, agents, heads, time,  time)
                mask = mask_x.transpose(1, 2)
                mask = mask.unsqueeze(2).repeat(1, 1, self.num_heads, 1)
                mask = mask.unsqueeze(-1).repeat(1, 1, 1, 1, A)
                att = att.masked_fill_(mask, self.mask_val)
            att = F.softmax(att, dim=-1)  # (T, H, A, N)
            att = self.attn_dropout(att)  # (T, H, A, N)

            # y shape: (batch_size, agents, heads, time, hidden_size / num_heads)
            y = att @ V
            # (batch_size, agents, heads, time, hidden_size / num_heads)->(batch_size, time, agents, heads, hidden_size)
            y = y.permute(0, 3, 1, 2, 4).contiguous().view(B, A, T, C) # (B, A, T, D=embed_size)
        else:
            error_message = f"Factorization axis {self.across} not supported."
            raise ValueError(error_message)

        # y shape: (batch_size, time, agents, hidden_size)
        y = self.c_proj(y)
        return self.resid_dropout(y)

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
            x[torch.Tensor]: input tensor to be self-attended.

        Output
        ------
            x[torch.Tensor]: attended output tensor.
        """
        x = self.ln_1(x)
        x = x + self.attn(x, mask_x=mask_x)
        x = self.ln_2(x)
        x = x + self.mlp(x, return_dict=False)
        return x


class SelfAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_channels: int,
        num_qk_channels: int | None = None,
        num_v_channels: int | None = None,
        max_heads_parallel: int | None = None,
        causal_attention: bool = False,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
    ):
        """Pre-layer norm self-attention (see `MultiHeadAttention` and for attention details)."""
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            num_q_input_channels=num_channels,
            num_kv_input_channels=num_channels,
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
        x: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        rot_pos_emb: RotaryPositionEmbedding | None = None,
        kv_cache: KVCache | None = None,
    ):
        """Pre-layer-norm self-attention of input `x`."""
        x = self.norm(x)
        return self.attention(
            x,
            x,
            pad_mask=pad_mask,
            rot_pos_emb_q=rot_pos_emb,
            rot_pos_emb_k=rot_pos_emb,
            kv_cache=kv_cache,
        )


class SelfAttentionLayer(AbstractAttentionLayer):
    def __init__(
        self,
        num_heads: int,
        num_channels: int,
        num_qk_channels: int | None = None,
        num_v_channels: int | None = None,
        max_heads_parallel: int | None = None,
        causal_attention: bool = False,
        widening_factor: int = 1,
        dropout: float = 0.0,
        residual_dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
        mlp_bias: bool = True,
    ):
        self_attn = SelfAttention(
            num_heads=num_heads,
            num_channels=num_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            max_heads_parallel=max_heads_parallel,
            causal_attention=causal_attention,
            dropout=dropout,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
        )

        self.num_qk_channels = self_attn.attention.num_qk_channels
        self.num_v_channels = self_attn.attention.num_v_channels

        super().__init__(
            Residual(self_attn, residual_dropout),
            Residual(MLP(num_channels, widening_factor, bias=mlp_bias), residual_dropout),
        )


class SelfAttentionBlock(nn.Sequential):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        num_channels: int,
        num_qk_channels: int | None = None,
        num_v_channels: int | None = None,
        num_rotary_layers: int = 1,
        max_heads_parallel: int | None = None,
        causal_attention: bool = False,
        widening_factor: int = 1,
        dropout: float = 0.0,
        residual_dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
        mlp_bias: bool = True,
    ):
        layers = [
            SelfAttentionLayer(
                num_heads=num_heads,
                num_channels=num_channels,
                num_qk_channels=num_qk_channels,
                num_v_channels=num_v_channels,
                max_heads_parallel=max_heads_parallel,
                causal_attention=causal_attention,
                widening_factor=widening_factor,
                dropout=dropout,
                residual_dropout=residual_dropout,
                qkv_bias=qkv_bias,
                out_bias=out_bias,
                mlp_bias=mlp_bias,
            )
            for _ in range(num_layers)
        ]

        self.num_rotary_layers = num_rotary_layers
        super().__init__(*layers)

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        rot_pos_emb: RotaryPositionEmbedding | None = None,
        kv_cache: list[KVCache] | None = None,
    ):
        if kv_cache is None:
            kv_cache_updated = None
        else:
            if len(kv_cache) == 0:
                # initialize kv_cache for each self-attention layer
                kv_cache = [layer.empty_kv_cache(x) for layer in self]
            kv_cache_updated = []

        for i, layer in enumerate(self):
            rot_pos_emb_use = i < self.num_rotary_layers or self.num_rotary_layers == -1
            rot_pos_emb_i = rot_pos_emb if rot_pos_emb_use else None

            kv_cache_i = None if kv_cache is None else kv_cache[i]
            output = layer(x, pad_mask=pad_mask, rot_pos_emb=rot_pos_emb_i, kv_cache=kv_cache_i)

            x = output.last_hidden_state

            if kv_cache_updated is not None:
                kv_cache_updated.append(output.kv_cache)

        return ModuleOutput(last_hidden_state=x, kv_cache=kv_cache_updated)
