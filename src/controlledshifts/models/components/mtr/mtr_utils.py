# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi
# All Rights Reserved

import torch
import torch.nn as nn


def get_batch_offsets(batch_idxs, bs):
    """
    Args:
        batch_idxs (N,): int tensor
        bs (int): batch size

    Returns:
        batch_offsets (bs + 1,): int tensor
    """
    batch_offsets = torch.zeros(bs + 1).int().to(batch_idxs.device)
    for i in range(bs):
        batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
    assert batch_offsets[-1] == batch_idxs.shape[0]
    return batch_offsets


def build_mlps(c_in, mlp_channels=None, ret_before_act=False, without_norm=False):
    layers = []
    num_layers = len(mlp_channels)

    for k in range(num_layers):
        if k + 1 == num_layers and ret_before_act:
            layers.append(nn.Linear(c_in, mlp_channels[k], bias=True))
        else:
            if without_norm:
                layers.extend([nn.Linear(c_in, mlp_channels[k], bias=True), nn.ReLU()])
            else:
                layers.extend(
                    [nn.Linear(c_in, mlp_channels[k], bias=False), nn.BatchNorm1d(mlp_channels[k]), nn.ReLU()])
            c_in = mlp_channels[k]

    return nn.Sequential(*layers)

@torch.no_grad()
def knn_batch(x_pos_stack, batch_offsets, k):
    """Pure-PyTorch KNN within each batch item.

    Replaces the original CUDA ``knn_batch_mlogk`` kernel.  For each valid token finds its K spatially nearest
    neighbours (by 2-D Euclidean distance) within the same batch item and returns their absolute indices into the flat
    ``x_pos_stack`` array — matching the index contract expected by ``MultiheadAttentionLocal``.

    Args:
        x_pos_stack (M, 3): positions of the M valid tokens, flat across batch
        batch_offsets (B+1,): cumulative token counts per batch item
        k (int): number of neighbours to return (including the token itself)

    Returns:
        index_pair (M, K): long tensor of absolute indices; columns beyond the batch-item size are filled with -1
        (masked in attention)
    """
    M = x_pos_stack.shape[0]
    B = batch_offsets.shape[0] - 1
    device = x_pos_stack.device
    index_pair = torch.full((M, k), -1, dtype=torch.long, device=device)

    for i in range(B):
        start = int(batch_offsets[i].item())
        end = int(batch_offsets[i + 1].item())
        if end <= start:
            continue
        pos_i = x_pos_stack[start:end, :2]          # (n_i, 2)
        k_actual = min(k, end - start)
        dists = torch.cdist(pos_i, pos_i)            # (n_i, n_i)
        _, nn_local = dists.topk(k_actual, dim=-1, largest=False)  # (n_i, k_actual)
        index_pair[start:end, :k_actual] = nn_local + start        # absolute indices

    return index_pair


class PointNetPolylineEncoder(nn.Module):
    """PointNet-based polyline encoder used in the MTR context encoder."""

    def __init__(self, in_channels, hidden_dim, num_layers=3, num_pre_layers=1, out_channels=None):
        super().__init__()
        self.pre_mlps = build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_dim] * num_pre_layers,
            ret_before_act=False
        )
        self.mlps = build_mlps(
            c_in=hidden_dim * 2,
            mlp_channels=[hidden_dim] * (num_layers - num_pre_layers),
            ret_before_act=False
        )

        if out_channels is not None:
            self.out_mlps = build_mlps(
                c_in=hidden_dim, mlp_channels=[hidden_dim, out_channels],
                ret_before_act=True, without_norm=True
            )
        else:
            self.out_mlps = None

    def forward(self, polylines, polylines_mask):
        """
        Args:
            polylines (batch_size, num_polylines, num_points_each_polyline, C)
            polylines_mask (batch_size, num_polylines, num_points_each_polyline): bool

        Returns:
            feature_buffers (batch_size, num_polylines, out_channels)
        """
        batch_size, num_polylines, num_points_each_polyline, C = polylines.shape

        # pre-mlp
        polylines_feature_valid = self.pre_mlps(polylines[polylines_mask])
        polylines_feature = polylines.new_zeros(
            batch_size, num_polylines, num_points_each_polyline,
            polylines_feature_valid.shape[-1]
        )
        polylines_feature[polylines_mask] = polylines_feature_valid

        # global max pool and concat
        pooled_feature = polylines_feature.max(dim=2)[0]
        polylines_feature = torch.cat(
            (polylines_feature, pooled_feature[:, :, None, :].repeat(1, 1, num_points_each_polyline, 1)), dim=-1
        )

        # mlp
        polylines_feature_valid = self.mlps(polylines_feature[polylines_mask])
        feature_buffers = polylines_feature.new_zeros(
            batch_size, num_polylines, num_points_each_polyline,
            polylines_feature_valid.shape[-1]
        )
        feature_buffers[polylines_mask] = polylines_feature_valid

        if self.out_mlps is not None:
            feature_buffers = feature_buffers.max(dim=2)[0]
            valid_mask = (polylines_mask.sum(dim=-1) > 0)
            feature_buffers_valid = self.out_mlps(feature_buffers[valid_mask])
            feature_buffers = feature_buffers.new_zeros(batch_size, num_polylines, feature_buffers_valid.shape[-1])
            feature_buffers[valid_mask] = feature_buffers_valid

        return feature_buffers
