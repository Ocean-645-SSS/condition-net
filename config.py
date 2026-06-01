"""ConditionNET 训练与模型配置"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # ── 模型结构 ──
    embed_dim: int = 384          # DINOv2-vits14 特征维度
    clip_dim: int = 512           # CLIP ViT-B/32 文本向量维度
    num_patches: int = 256        # 224/14 × 224/14
    num_heads: int = 6            # 多头注意力头数
    num_layers: int = 4           # Transformer 编码器层数

    # ── 训练参数 ──
    batch_size: int = 32
    epochs: int = 40
    peak_lr: float = 5e-4         # ADAMW 峰值学习率
    weight_decay: float = 0.2
    beta1: float = 0.9
    beta2: float = 0.98
    warmup_epochs: int = 4        # 余弦退火的线性预热阶段

    # ── 损失函数 ──
    temperature: float = 0.07     # InfoNCE 温度系数 τ

    # ── 数据 ──
    img_size: int = 224
    train_val_split: float = 0.7
    num_paraphrases: int = 20     # 每条指令的改写变体数

    # ── 音频模态 ──
    use_audio: bool = True         # 是否启用音频模态
    audio_sample_rate: int = 16000
    audio_duration: float = 1.0    # 音频片段时长 (秒)
    audio_n_mels: int = 128

    # ── 路径 ──
    data_root: str = "data"
    output_dir: str = "checkpoints"
    log_dir: str = "logs"

    # ── ImperfectPour 数据集路径 ──
    imperfectpour_data_dir: str = "data/imperfectpour"
    imperfectpour_train_csv: str = "data/imperfectpour/train.csv"
    imperfectpour_val_csv: str = "data/imperfectpour/val.csv"
    imperfectpour_img_dir: str = "data/imperfectpour/frames"

    # ── 设备 ──
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
