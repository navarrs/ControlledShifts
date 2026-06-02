# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi
# All Rights Reserved

import math

import torch


def gen_sineembed_for_position(pos_tensor, hidden_dim=256):
    """Generate sinusoidal position embeddings.

    Args:
        pos_tensor (shape: (n, b, 2)): x, y positions
        hidden_dim (int): output embedding dimension

    Returns:
        pos_embed (shape: (n, b, hidden_dim))
    """
    scale = 2 * math.pi
    dim_t = torch.arange(hidden_dim // 2, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / (hidden_dim // 2))
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_x, pos_y), dim=2)
    return pos
