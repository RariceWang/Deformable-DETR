# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from .ops.modules import MSDeformAttn


class ReAttentionModule(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_levels=4, n_points=4, d_ffn=1024, dropout=0.1, activation="relu"):
        super().__init__()
        
        # Cross Attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def forward(self, query, reference_points, src_flatten, src_spatial_shapes, src_level_start_index, src_padding_mask=None):
        """
        Args:
            query: (bs, num_queries, d_model)
            reference_points: (bs, num_queries, n_levels, 2) or (bs, num_queries, n_levels, 4)
            src_flatten: (bs, num_keys, d_model)
            src_spatial_shapes: (n_levels, 2)
            src_level_start_index: (n_levels, )
            src_padding_mask: (bs, num_keys)
        """
        # MSDeformAttn expects reference_points to be (N, Length_{query}, n_levels, 2 or 4)
        # If we pass (bs, num_queries, 4), we need to expand it to n_levels
        
        bs, num_queries, _ = query.shape
        if reference_points.dim() == 3:
             # (bs, num_queries, 4) -> (bs, num_queries, n_levels, 4)
             reference_points = reference_points.unsqueeze(2).repeat(1, 1, self.cross_attn.n_levels, 1)
        
        src2 = self.cross_attn(query, reference_points, src_flatten, src_spatial_shapes, src_level_start_index, src_padding_mask)
        query = query + self.dropout1(src2)
        query = self.norm1(query)
        
        # FFN
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(query))))
        query = query + self.dropout3(src2)
        query = self.norm2(query)
        
        return query


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
