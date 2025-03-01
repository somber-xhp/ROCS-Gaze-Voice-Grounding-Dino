# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .utils import (
    MLP,
    _get_activation_fn,
    _get_clones,
    gen_encoder_output_proposals,
    gen_sineembed_for_position,
    sigmoid_focal_loss,
)


class TextTransformer(nn.Module):
    def __init__(self, num_layers, d_model=256, nheads=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.nheads = nheads
        self.dim_feedforward = dim_feedforward
        self.norm = None

        single_encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nheads, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.layers = _get_clones(single_encoder_layer, num_layers)

    def forward(self, memory_text: torch.Tensor, text_attention_mask: torch.Tensor):
        """

        Args:
            text_attention_mask: bs, num_token
            memory_text: bs, num_token, d_model

        Raises:
            RuntimeError: _description_

        Returns:
            output: bs, num_token, d_model
        """

        output = memory_text.transpose(0, 1)

        for layer in self.layers:
            output = layer(output, src_key_padding_mask=text_attention_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output.transpose(0, 1)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.nhead = nhead

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        # repeat attn mask
        if src_mask.dim() == 3 and src_mask.shape[0] == src.shape[1]:
            # bs, num_q, num_k
            src_mask = src_mask.repeat(self.nhead, 1, 1)

        q = k = self.with_pos_embed(src, pos)

        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask)[0]

        # src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

#2.7
class QuerySelector(nn.Module):
    def __init__(self, d_model, nhead):
        super(QuerySelector, self).__init__()
        self.cross_attention = nn.MultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, memory, mask_flatten, spatial_shapes, text_dict, gaze_features):
        # 拼接热力图特征与原始图片特征
        combined_features = torch.cat((memory, gaze_features), dim=-1)

        # 将文本特征和拼接特征融合
        text_features = text_dict["bert_output"]["last_hidden_state"]
        attn_output, _ = self.cross_attention(combined_features, text_features, text_features)

        # 语言引导的查询选择
        refpoint_embed, tgt, init_box_proposal = self.language_guided_query_selection(
            attn_output, text_dict, mask_flatten, spatial_shapes
        )
        
        return refpoint_embed, tgt, init_box_proposal

    def language_guided_query_selection(self, attn_output, text_dict, mask_flatten, spatial_shapes):
        # 初始化查询
        batch_size, num_queries, _ = attn_output.size()
        refpoint_embed = self.linear1(attn_output)
        refpoint_embed = F.relu(refpoint_embed)
        refpoint_embed = self.dropout(refpoint_embed)
        refpoint_embed = self.linear2(refpoint_embed)
        refpoint_embed = self.norm1(refpoint_embed)

        # 初始化目标
        tgt = self.linear1(attn_output)
        tgt = F.relu(tgt)
        tgt = self.dropout(tgt)
        tgt = self.linear2(tgt)
        tgt = self.norm2(tgt)

        # 初始化边界框建议
        init_box_proposal = torch.zeros((batch_size, num_queries, 4), device=attn_output.device)
        for b in range(batch_size):
            for q in range(num_queries):
                x_center = torch.sigmoid(attn_output[b, q, 0])
                y_center = torch.sigmoid(attn_output[b, q, 1])
                width = torch.sigmoid(attn_output[b, q, 2])
                height = torch.sigmoid(attn_output[b, q, 3])
                init_box_proposal[b, q, 0] = x_center - width / 2
                init_box_proposal[b, q, 1] = y_center - height / 2
                init_box_proposal[b, q, 2] = x_center + width / 2
                init_box_proposal[b, q, 3] = y_center + height / 2

        return refpoint_embed, tgt, init_box_proposal


class ModifiedModel(nn.Module):
    def __init__(self, d_model, nhead, backbone, transformer):
        super(ModifiedModel, self).__init__()
        self.feature_enhancer_with_gaze = FeatureEnhancerWithGaze(d_model, nhead)
        self.query_selector = QuerySelector(d_model, nhead)
        self.gaze_cross_attention = nn.MultiheadAttention(d_model, nhead)
        self.backbone = backbone  # 传入实际的backbone实例
        self.transformer = transformer  # 传入实际的transformer实例

    def forward(self, tensor_list, text_dict, gaze_features):
        # 从 backbone 获取图像特征
        features, pos = self.backbone(tensor_list)
        image_features = features[-1].tensors
        
        # 应用特征融合模块
        image_features, gaze_features = self.feature_enhancer_with_gaze(image_features, gaze_features)
        
        # Gaze Cross-Attention
        image_features, _ = self.gaze_cross_attention(image_features, gaze_features, gaze_features)
        
        # 获取 mask 和 spatial_shapes
        mask_flatten = features[-1].mask.flatten(1)
        spatial_shapes = torch.as_tensor([list(f.tensors.shape[-2:]) for f in features], dtype=torch.long, device=image_features.device)
        
        # 应用查询选择器
        refpoint_embed, tgt, init_box_proposal = self.query_selector(
            image_features, mask_flatten, spatial_shapes, text_dict, gaze_features
        )

        # 其他前向传播步骤
        transformed_features = self.transformer(refpoint_embed, tgt)
        return transformed_features
