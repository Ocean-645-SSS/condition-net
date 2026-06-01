"""
ConditionNET 第一层：提取环境全局语义特征（支持多模态 token）

序列结构:
  [visual_CLS, audio_token, patch_1, ..., patch_256]
  共计 258 个 token (无音频时为 257)

参数:
  embed_dim:   视觉特征维度 (DINOv2-vits14 默认为 384)
  num_patches: 图像切块数量 (224x224 图像 / 14x14 patch = 256)
  num_heads:   多头注意力的头数
  num_layers:   Transformer 编码器的层数
"""

import torch
import torch.nn as nn


class StateTransformer(nn.Module):
    def __init__(self, embed_dim=384, num_patches=256, num_heads=6, num_layers=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        # total_tokens = CLS + audio + patches，预留 audio 位置
        self.total_tokens = 2 + num_patches  # 1 CLS + 1 audio + 256 patches = 258

        # 1. 可学习的 CLS Token (全局视觉状态)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # 2. 位置编码: [CLS, audio, patch_1, ..., patch_256]，共 258 个
        self.pos_embed = nn.Parameter(torch.randn(1, self.total_tokens, embed_dim))

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, visual_tokens, audio_token=None):
        """
        输入:
            visual_tokens: DINOv2 特征 [B, 256, 384]
            audio_token:   音频特征 [B, 1, 384] 或 None (纯视觉模式)
        输出:
            state_features: [B, total_tokens, 384]
        """
        B = visual_tokens.shape[0]

        # CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)          # [B, 1, 384]

        if audio_token is not None:
            # 多模态: [CLS, audio, 256 patches] → 258 tokens
            x = torch.cat((cls_tokens, audio_token, visual_tokens), dim=1)
            x = x + self.pos_embed
        else:
            # 纯视觉: [CLS, 256 patches] → 257 tokens (跳过 audio 位置)
            x = torch.cat((cls_tokens, visual_tokens), dim=1)
            pos_no_audio = torch.cat([
                self.pos_embed[:, :1, :],                      # CLS position
                self.pos_embed[:, 2:, :],                       # skip audio pos
            ], dim=1)
            x = x + pos_no_audio

        state_features = self.transformer(x)
        return state_features
