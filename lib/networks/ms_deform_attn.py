# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_



def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=768, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads

        self.im2col_step = 64

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_points * 3)
        self.attention_weights = nn.Linear(d_model, n_heads * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos() * thetas.cos(), thetas.cos() * thetas.sin(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 3).repeat(1, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def ms_deform_attn_core_pytorch(self, value, sampling_locations, attention_weights):
        # for debug and test only,
        # need to use cuda version instead
        N_, S_, M_, D_ = value.shape  # M_: Multi-head; D_: sampling-points * 2
        _, Lq_, M_, P_, _ = sampling_locations.shape
        sampling_grids = 2 * sampling_locations - 1  # grid 中 xy 的取值范围在[-1,1]
        n = 12
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value = value.flatten(2).transpose(1, 2).reshape(N_ * M_, D_, n, n, n)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grids = sampling_grids[:, :, :].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_ (grid_sample: input:(N,C,H_in,W_in); grid:(N,H_out,W_out,2); output:(N,C,H_out,W_out))
        sampling_value = F.grid_sample(value, sampling_grids.unsqueeze(2),
                                              mode='bilinear', padding_mode='zeros', align_corners=False)
        # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_ * M_, 1, Lq_, L_*P_)
        attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, P_)
        output = (sampling_value.flatten(-2) * attention_weights).sum(-1).view(N_, M_ * D_, Lq_)
        return output.transpose(1, 2).contiguous()


    def forward(self, query, reference_points, input_flatten, input_spatial_shapes=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape

        value = self.value_proj(input_flatten)
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_points, 3)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_points)
        # N, Len_q, n_heads, n_points, 3
        n = 12
        spatial_shapes = torch.as_tensor((n, n, n), dtype=torch.long, device=value.device)
        offset_normalizer = torch.stack([spatial_shapes[2], spatial_shapes[1], spatial_shapes[0]], -1)
        sampling_locations = reference_points[:, :, None, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, None, :]

        output = self.ms_deform_attn_core_pytorch(value, sampling_locations, attention_weights)
        # output = MSDeformAttnFunction.apply(
        #     value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights,
        #     self.im2col_step)
        output = self.output_proj(output)
        return output
