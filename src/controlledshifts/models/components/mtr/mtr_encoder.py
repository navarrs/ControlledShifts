# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi
# All Rights Reserved

import torch
import torch.nn as nn

from .mtr_utils import PointNetPolylineEncoder, get_batch_offsets, knn_batch
from .transformer import position_encoding_utils, transformer_encoder_layer


class MTREncoder(nn.Module):
    """Context encoder for the Motion Transformer.

    Encodes agent trajectories and map polylines via PointNet-based polyline
    encoders followed by global (or local) transformer self-attention.
    """

    def __init__(self, config):
        """
        Args:
            config (dict/DictConfig): encoder configuration with keys:
                num_input_attr_agent (int): agent feature dimension (excl. mask)
                num_input_attr_map (int): map feature dimension
                num_channel_in_mlp_agent (int): hidden dim for agent MLP
                num_layer_in_mlp_agent (int): number of MLP layers for agent encoder
                num_channel_in_mlp_map (int): hidden dim for map MLP
                num_layer_in_mlp_map (int): number of MLP layers for map encoder
                num_layer_in_pre_mlp_map (int): number of pre-MLP layers for map
                d_model (int): transformer hidden dimension
                num_attn_layers (int): number of transformer self-attention layers
                num_attn_head (int): number of attention heads
                dropout_of_attn (float): dropout probability in attention
                use_local_attn (bool): use local KNN attention (requires CUDA ops)
                num_of_attn_neighbors (int): K for local KNN attention
        """
        super().__init__()
        self.model_cfg = config

        # Polyline encoders
        self.agent_polyline_encoder = PointNetPolylineEncoder(
            in_channels=self.model_cfg['num_input_attr_agent'] + 1,  # +1 for validity mask
            hidden_dim=self.model_cfg['num_channel_in_mlp_agent'],
            num_layers=self.model_cfg['num_layer_in_mlp_agent'],
            out_channels=self.model_cfg['d_model']
        )
        self.map_polyline_encoder = PointNetPolylineEncoder(
            in_channels=self.model_cfg['num_input_attr_map'],
            hidden_dim=self.model_cfg['num_channel_in_mlp_map'],
            num_layers=self.model_cfg['num_layer_in_mlp_map'],
            num_pre_layers=self.model_cfg['num_layer_in_pre_mlp_map'],
            out_channels=self.model_cfg['d_model']
        )

        # Transformer self-attention layers
        self.use_local_attn = self.model_cfg.get('use_local_attn', False)

        self_attn_layers = []
        for _ in range(self.model_cfg['num_attn_layers']):
            self_attn_layers.append(
                transformer_encoder_layer.TransformerEncoderLayer(
                    d_model=self.model_cfg['d_model'],
                    nhead=self.model_cfg['num_attn_head'],
                    dim_feedforward=self.model_cfg['d_model'] * 4,
                    dropout=self.model_cfg.get('dropout_of_attn', 0.1),
                    normalize_before=False,
                    use_local_attn=self.use_local_attn,
                )
            )
        self.self_attn_layers = nn.ModuleList(self_attn_layers)
        self.num_out_channels = self.model_cfg['d_model']

    def apply_local_attn(self, x: torch.Tensor, x_mask: torch.Tensor, x_pos: torch.Tensor, num_of_neighbors: int) -> torch.Tensor:
        """Local self-attention over K spatial nearest neighbours.

        Args:
            x (B, N, d_model)
            x_mask (B, N): True = valid token
            x_pos (B, N, 3): XYZ positions for KNN lookup
            num_of_neighbors (int): K
        """
        assert torch.all(x_mask.sum(dim=-1) > 0), \
            "Every sample must have at least one valid token."

        batch_size, N, d_model = x.shape

        x_stack_full = x.view(-1, d_model)             # (B*N, d_model)
        x_mask_stack = x_mask.view(-1)                 # (B*N,)
        x_pos_stack_full = x_pos.view(-1, 3)

        batch_idxs_full = torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, N).reshape(-1).int()

        # Filter to valid tokens only
        x_stack = x_stack_full[x_mask_stack]           # (M, d_model)
        x_pos_stack = x_pos_stack_full[x_mask_stack]   # (M, 3)
        batch_idxs = batch_idxs_full[x_mask_stack]     # (M,)

        batch_offsets = get_batch_offsets(batch_idxs, batch_size)  # (B+1,)
        batch_cnt = batch_offsets[1:] - batch_offsets[:-1]         # (B,)

        index_pair = knn_batch(x_pos_stack, batch_offsets, num_of_neighbors)  # (M, K)

        pos_embedding = position_encoding_utils.gen_sineembed_for_position(
            x_pos_stack[None, :, 0:2], hidden_dim=d_model
        )[0]  # (M, d_model)

        output = x_stack
        for layer in self.self_attn_layers:
            output = layer(
                src=output,
                pos=pos_embedding,
                index_pair=index_pair,
                query_batch_cnt=batch_cnt,
                key_batch_cnt=batch_cnt,
                index_pair_batch=batch_idxs,
            )

        ret_full = torch.zeros_like(x_stack_full)
        ret_full[x_mask_stack] = output
        return ret_full.view(batch_size, N, d_model)

    def apply_global_attn(self, x, x_mask, x_pos):
        """Global self-attention over all tokens.

        Args:
            x (B, N, d_model)
            x_mask (B, N): True = valid token
            x_pos (B, N, 3): XYZ positions for positional encoding
        """
        assert torch.all(x_mask.sum(dim=-1) > 0), \
            "Every sample must have at least one valid token."

        batch_size, N, d_model = x.shape
        # (N, B, d_model) layout expected by the encoder layers
        x_t = x.permute(1, 0, 2)
        x_pos_t = x_pos.permute(1, 0, 2)  # (N, B, 3)

        pos_embedding = position_encoding_utils.gen_sineembed_for_position(
            x_pos_t, hidden_dim=d_model
        )  # (N, B, d_model)

        # x_mask is (B, N) — correct shape for key_padding_mask (True = ignore)
        src_key_padding_mask = ~x_mask  # True means "masked out"

        for layer in self.self_attn_layers:
            x_t = layer(
                src=x_t,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos_embedding
            )

        return x_t.permute(1, 0, 2)  # (B, N, d_model)

    def forward(self, batch_dict):
        """Encode agent and map context into per-token features.

        Args:
            batch_dict (dict): must contain ``input_dict`` with:
                obj_trajs (B, N, H, Da)
                obj_trajs_mask (B, N, H): bool
                map_polylines (B, P, L, Dm)
                map_polylines_mask (B, P, L): bool
                obj_trajs_last_pos (B, N, 3)
                map_polylines_center (B, P, 3)
                track_index_to_predict (B,): index of the ego agent

        Adds to batch_dict:
            center_objects_feature (B, d_model)
            obj_feature (B, N, d_model)
            map_feature (B, P, d_model)
            obj_mask (B, N)
            map_mask (B, P)
            obj_pos (B, N, 3)
            map_pos (B, P, 3)
        """
        input_dict = batch_dict['input_dict']
        obj_trajs = input_dict['obj_trajs']
        obj_trajs_mask = input_dict['obj_trajs_mask']
        map_polylines = input_dict['map_polylines']
        map_polylines_mask = input_dict['map_polylines_mask']
        obj_trajs_last_pos = input_dict['obj_trajs_last_pos']
        map_polylines_center = input_dict['map_polylines_center']
        track_index_to_predict = input_dict['track_index_to_predict'].long()

        assert obj_trajs_mask.dtype == torch.bool and map_polylines_mask.dtype == torch.bool

        num_center_objects, num_objects, num_timestamps, _ = obj_trajs.shape
        num_polylines = map_polylines.shape[1]

        # Append validity mask as last feature channel for the agent encoder
        obj_trajs_in = torch.cat(
            (obj_trajs, obj_trajs_mask[:, :, :, None].type_as(obj_trajs)), dim=-1
        )

        obj_polylines_feature = self.agent_polyline_encoder(
            obj_trajs_in, obj_trajs_mask
        )  # (B, N, d_model)
        map_polylines_feature = self.map_polyline_encoder(
            map_polylines, map_polylines_mask
        )  # (B, P, d_model)

        obj_valid_mask = (obj_trajs_mask.sum(dim=-1) > 0)    # (B, N)
        map_valid_mask = (map_polylines_mask.sum(dim=-1) > 0)  # (B, P)

        # Concatenate agent and map tokens for joint attention
        global_token_feature = torch.cat((obj_polylines_feature, map_polylines_feature), dim=1)
        global_token_mask = torch.cat((obj_valid_mask, map_valid_mask), dim=1)
        global_token_pos = torch.cat((obj_trajs_last_pos, map_polylines_center), dim=1)

        if self.use_local_attn:
            global_token_feature = self.apply_local_attn(
                x=global_token_feature, x_mask=global_token_mask, x_pos=global_token_pos,
                num_of_neighbors=self.model_cfg['num_of_attn_neighbors'],
            )
        else:
            global_token_feature = self.apply_global_attn(
                x=global_token_feature, x_mask=global_token_mask, x_pos=global_token_pos
            )

        obj_polylines_feature = global_token_feature[:, :num_objects]
        map_polylines_feature = global_token_feature[:, num_objects:]
        assert map_polylines_feature.shape[1] == num_polylines

        # Extract the feature of the center (ego) object
        center_objects_feature = obj_polylines_feature[
            torch.arange(num_center_objects), track_index_to_predict
        ]

        batch_dict['center_objects_feature'] = center_objects_feature
        batch_dict['obj_feature'] = obj_polylines_feature
        batch_dict['map_feature'] = map_polylines_feature
        batch_dict['obj_mask'] = obj_valid_mask
        batch_dict['map_mask'] = map_valid_mask
        batch_dict['obj_pos'] = obj_trajs_last_pos
        batch_dict['map_pos'] = map_polylines_center

        return batch_dict
