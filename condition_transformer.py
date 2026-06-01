import torch
import torch.nn as nn


class ConditionTransformer(nn.Module):
    """
        ConditionNET 的第二层：动作语义与视觉特征对齐
        参数:
            embed_dim: 视觉特征维度 (384)
            clip_dim: CLIP 文本向量维度 (通常为 512)
            num_head:多头自注意力机制的头数
            num_layers:transformer编码器堆叠层数
        """
    def __init__(self, embed_dim=384, clip_dim=512, num_heads=6, num_layers=4):  
        super().__init__()

        # 1. 维度对齐层 
        # CLIP 的维度(512)通常与 DINOv2(384) 不同，必须映射对齐
        self.text_proj = nn.Linear(clip_dim, embed_dim)

        # 2. Transformer Encoder  ,实现 Cross-Attention 效果
        # 这里的 Self-Attention 因为包含了指令 Token，实际上起到了交叉注意力的作用
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, #输入/输出特征维度
            nhead=num_heads, #6个注意力头并行计算
            dim_feedforward=embed_dim * 4, #隐藏层维度
            batch_first=True, #输入输出张量格式为[Batch,Seq,Dim]
            activation='gelu' #gelu激活函数比relu更平滑
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. 最终分类预测头 (MLP Head)
        # 输出 3 个类别的 Logits: Precondition, Effect, Unsatisfied
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, 256), #384->256维
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3) #256->3类别
        )

    def forward(self, state_features, text_embedding):
        """
        前向传播
        输入:
            state_features: 上一层 StateTransformer 的完整输出 [B, 257, 384]
            text_embedding: CLIP 提取的指令特征 [B, 512]
        输出:
            logits: 三分类概率得分 [B, 3]
            E: 最终融合后的 condition 特征 (用于 Loss 计算) [B, 384]
        """
        B = state_features.shape[0] # 获得当前批大小

        # 1. 对齐文本特征维度: [B, 512] -> [B, 1, 384]
        s = self.text_proj(text_embedding).unsqueeze(1)

        # 2. 核心！：替换 Token 
        # state_features[:, 0, :] 是上一层的 CLS
        # 我们丢弃它，换成指令向量 s，保持剩下的 256 个视觉 Patch 不变
        x = torch.cat((s, state_features[:, 1:, :]), dim=1)  # [B, 257, 384]

        # 3. 执行 Condition 变换 (多模态交互)
        # 此时指令向量 s 会通过 Attention 机制“检索”视觉 Patch
        output = self.transformer(x)

        # 4. 提取最终的特征向量 E (位于序列第 0 位)
        E = output[:, 0, :]  # [B, 384]

        # 5. 分类预测
        logits = self.mlp_head(E)

        return logits, E