"""
ConditionNET 损失函数
  L_condition: 3 类 CrossEntropy (precondition / effect / unsatisfied)
  L_consistency: InfoNCE (ea = cls⁺ - cls⁻) ↔ paraphrased action sp
  仅对成功执行的 demonstration 计算 consistency loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """批次内 InfoNCE: 最大化匹配对的相似度，最小化非匹配对"""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, ea, sp):
        """
        ea: 状态变化向量 [B, D]  (cls⁺ - cls⁻)
        sp: 改写动作文本向量 [B, D]
        返回: InfoNCE 标量损失
        """
        B = ea.shape[0]

        # L2 归一化
        ea = F.normalize(ea, dim=-1)
        sp = F.normalize(sp, dim=-1)

        # 相似度矩阵 [B, B]: logits[i][j] = sim(ea_i, sp_j) / τ
        logits = ea @ sp.T / self.temperature

        # 标签: 对角线位置是正样本
        labels = torch.arange(B, device=ea.device)

        # 双向 InfoNCE
        loss_i2t = F.cross_entropy(logits, labels)       # 图像→文本
        loss_t2i = F.cross_entropy(logits.T, labels)     # 文本→图像

        return (loss_i2t + loss_t2i) / 2


class ConditionLoss(nn.Module):
    """组合损失函数，含首 batch 权重归一化"""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.infonce = InfoNCELoss(temperature)

        # 文本投影: CLIP dim (512) → visual dim (384)
        self.text_proj_consistency = nn.Linear(512, 384)

        # 自适应权重 (首 batch 后固定)
        self.alpha: float = 0.5
        self.beta: float = 0.5
        self.weights_locked: bool = False
        self.lock_alpha: float = 0.5
        self.lock_beta: float = 0.5

    def forward(self, logits, labels, cls_pre, cls_post, text_paraphrase, success_mask):
        """
        logits:          分类预测 [B, 3]
        labels:          分类标签 [B]  (0=precondition, 1=effect, 2=unsatisfied)
        cls_pre:         I⁻ 的 state CLS [B, 384]
        cls_post:        I⁺ 的 state CLS [B, 384]
        text_paraphrase: 改写动作文本的 CLIP 向量 [B, 512]
        success_mask:    布尔掩码 [B]，True 表示成功 demonstration
        返回: total_loss, l_cond, l_cons
        """
        # L_condition: 3 类交叉熵
        l_cond = self.ce_loss(logits, labels)

        # L_consistency: 仅对成功 demonstration
        mask = success_mask.bool()
        if mask.sum() >= 2:  # InfoNCE 至少需要 2 个样本
            ea = cls_post[mask] - cls_pre[mask]                     # [N, 384]
            sp = self.text_proj_consistency(text_paraphrase[mask])  # [N, 384]
            l_cons = self.infonce(ea, sp)
        else:
            l_cons = torch.tensor(0.0, device=l_cond.device)

        # 首 batch 锁定权重
        if not self.weights_locked:
            self._lock_weights(l_cond, l_cons)

        total = self.lock_alpha * l_cond + self.lock_beta * l_cons
        return total, l_cond.detach(), l_cons.detach() if isinstance(l_cons, torch.Tensor) else l_cons

    @torch.no_grad()
    def _lock_weights(self, l_cond, l_cons):
        """首 batch 后动态平衡两个损失项的量级"""

        # 安全处理: 避免除零
        c = l_cond.item() if isinstance(l_cond, torch.Tensor) else l_cond
        s = l_cons.item() if isinstance(l_cons, torch.Tensor) and l_cons > 0 else l_cons

        # 逻辑: 量级较小的损失项配更大的权重，使两者贡献相当
        if isinstance(s, float) and s > 0:
            total = c + s
            # α ∝ 1/c 的倒数归一化 → 简化等价于 α·c ≈ β·s
            # → α = s/(c+s), β = c/(c+s)
            self.lock_alpha = s / total
            self.lock_beta = c / total
        else:
            self.lock_alpha = 1.0
            self.lock_beta = 0.0

        self.weights_locked = True
