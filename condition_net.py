"""
ConditionNET 完整架构组装（支持视觉+音频+文本三模态）

序列结构:
  State Transformer:     [vis_CLS, audio_token, 256 patches] → 258 tokens
  Condition Transformer: [text_proj, audio_token, 256 patches] → 258 tokens

参数:
  embed_dim: 视觉特征维度 (DINOv2-vits14 对应 384)
  clip_dim:  CLIP 文本向量维度 (通常为 512)
  num_heads: 多头注意力的头数
  num_layers: Transformer 编码器的层数
"""

import torch
import torch.nn as nn
from state_transformer import StateTransformer
from condition_transformer import ConditionTransformer


class ConditionNET(nn.Module):
    def __init__(self, embed_dim=384, clip_dim=512, num_heads=6, num_layers=4):
        super().__init__()

        # 1. State Transformer (环境全局语义)
        self.state_transformer = StateTransformer(
            embed_dim=embed_dim,
            num_patches=256,
            num_heads=num_heads,
            num_layers=num_layers,
        )

        # 2. Condition Transformer (指令对齐与分类)
        self.condition_transformer = ConditionTransformer(
            embed_dim=embed_dim,
            clip_dim=clip_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )

    def forward(self, visual_tokens, text_embedding, audio_token=None):
        """
        输入:
            visual_tokens:  DINOv2 patch 特征 [B, 256, 384]
            text_embedding:  CLIP 文本向量    [B, 512]
            audio_token:    音频特征 [B, 1, 384] 或 None (纯视觉/文本模式)
        输出:
            logits:    [B, 3]    三类 Logits
            E:         [B, 384]  融合指令后的特征 (用于 Consistency Loss)
            state_cls: [B, 384]  纯视觉全局特征 (用于 Consistency Loss)
        """

        # 1: 多模态环境编码
        state_out = self.state_transformer(visual_tokens, audio_token=audio_token)
        # state_out: [B, 258, 384] (有音频) 或 [B, 257, 384] (无音频)

        # 视觉 CLS 始终在位置 0
        state_cls = state_out[:, 0, :]   # [B, 384]

        # 2: 指令注入 → Token Replacement
        # 用 text 替换位置 0 的 vis_CLS，audio 和 patches 不变
        logits, E = self.condition_transformer(state_out, text_embedding)

        return logits, E, state_cls
