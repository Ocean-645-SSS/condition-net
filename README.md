# ConditionNET

<div align="center">

**基于多模态 Transformer 的机器人操作条件关系预测**

*环境状态 + 动作指令 → 条件关系分类（前置条件 / 效果 / 不满足）*

</div>

---

## 📖 简介

ConditionNET 是一个用于**机器人操作条件推理**的深度学习模型。给定机器人操作前后的环境图像（可选音频）和一条自然语言动作指令，模型判断该动作与环境之间的条件关系属于以下三类之一：

| 类别 | 含义 | 示例 |
|------|------|------|
| **Precondition** (0) | 场景展示了动作执行前的状态 | 杯子是空的，即将倒入牛奶 |
| **Effect** (1) | 场景展示了动作执行后的结果 | 牛奶已倒入杯中 |
| **Unsatisfied** (2) | 场景与动作描述不匹配 | "倒牛奶"的指令，但图中在切菜 |

### 核心设计

- **双塔 Transformer 架构**：State Transformer 提取环境全局语义，Condition Transformer 通过 Token Replacement 机制将文本指令注入视觉特征序列
- **多模态支持**：同时处理视觉（DINOv2）、文本（CLIP）、音频（Mel-spectrogram + CNN）三种模态
- **复合损失函数**：L = α·L_condition + β·L_consistency，在分类损失基础上加入 InfoNCE 一致性约束
- **冻结 Backbone + 可训练 Transformer**：DINOv2 和 CLIP 权重冻结，仅训练轻量的 Transformer 层和 Audio Encoder，高效利用预训练特征

---

## 🏗️ 模型架构

```
[DINOv2-vits14]          [CLIP ViT-B/32]        [Audio Encoder]
      ↓                       ↓                       ↓
 Patch Tokens            Text Embedding          Audio Token
 [B,256,384]               [B,512]               [B,1,384]
      ↓                       ↓                       ↓
 ┌────────────────────────────────────────────────────────┐
 │  State Transformer                                      │
 │  [vis_CLS | audio | 256 patches] → [B, 258, 384]      │
 │  提取环境全局语义                                       │
 └──────────────────────┬─────────────────────────────────┘
                        ↓
 ┌────────────────────────────────────────────────────────┐
 │  Condition Transformer                                  │
 │  用 text_proj 替换 vis_CLS → Cross-Attention            │
 │  [text | audio | 256 patches] → [B, 258, 384]          │
 │  指令与视觉对齐                                         │
 └──────────────────────┬─────────────────────────────────┘
                        ↓
                  MLP Head (384→256→3)
                        ↓
              [Precondition | Effect | Unsatisfied]
```

**Token Replacement** 是架构的核心：Condition Transformer 将 State Transformer 输出的 CLS token 替换为经过维度对齐的文本向量，使文本指令通过 Self-Attention 在视觉 token 序列上进行「软检索」，实现跨模态交互。

---

## 📂 项目结构

```
condition-net/
├── src/
│   ├── main.py                      # 入口脚本（训练/评估/推理）
│   ├── config.py                    # 超参数配置
│   ├── condition_net.py             # ConditionNET 完整模型组装
│   ├── state_transformer.py         # State Transformer 层
│   ├── condition_transformer.py     # Condition Transformer 层
│   ├── backbones.py                 # DINOv2 + CLIP 冻结编码器
│   ├── audio_encoder.py             # 音频 Mel-CNN 编码器
│   ├── dataset.py                   # 数据集加载与增强
│   ├── losses.py                    # 复合损失函数
│   ├── trainer.py                   # 训练器（ADAMW + 预热 + 余弦退火）
│   ├── run_imperfectpour.py         # ImperfectPour 一键训练脚本
│   ├── preprocess_imperfectpour.py  # 数据预处理（视频→帧+音频+CSV）
│   ├── setup_autodl.sh              # AutoDL 环境配置脚本

```

---

## 🚀 快速开始

### 环境要求

- Python ≥ 3.9
- PyTorch ≥ 2.0.0
- CUDA（推荐，CPU 也可运行）
- ffmpeg（数据预处理需要）

### 1. 安装依赖

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers pillow tqdm numpy
```

### 2. 准备数据集

#### 使用 ImperfectPour 数据集（推荐）

```bash
# 解压数据集
unzip ImperfectPour.zip -d /path/to/data/

# 一键运行：自动提取帧+音频+训练
python src/run_imperfectpour.py --dataset_root /path/to/ImperfectPour/ImperfectPour
```

#### 使用自定义数据集

按以下 CSV 格式准备数据（支持 3 列或 5 列）：

```csv
pre_image_path,post_image_path,pre_audio_path,post_audio_path,action_text
demo1/pre_00001.png,demo1/post_00001.png,demo1/audio_00001.wav,demo1/audio_00001.wav,pour milk
...
```

```bash
python src/main.py --mode train \
    --dataset default \
    --data_csv data/my_dataset.csv \
    --img_dir data/images \
    --epochs 40 \
    --batch_size 32
```

### 3. 训练

```bash
# ImperfectPour 数据集
python src/run_imperfectpour.py --epochs 40 --batch_size 16

# 通用数据集
python src/main.py --mode train \
    --dataset default \
    --data_csv data/demos.csv \
    --img_dir data/images

# 从检查点恢复训练
python src/main.py --mode train --ckpt checkpoints/best_model.pt
```

### 4. 评估

```bash
python src/main.py --mode eval \
    --data_csv data/demos.csv \
    --img_dir data/images \
    --ckpt checkpoints/best_model.pt
```

输出示例：
```
[Eval] 总准确率: 85.23% (512/601)
  precondition: 87.12%
  effect: 83.45%
  unsatisfied: 85.10%
```

### 5. 单张推理

```bash
python src/main.py --mode infer \
    --image path/to/image.jpg \
    --text "pick up the cup" \
    --ckpt checkpoints/best_model.pt
```

---

## ⚙️ 配置说明

所有超参数集中在 `src/config.py` 中，可通过命令行参数覆盖常用选项：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `embed_dim` | 384 | DINOv2-vits14 特征维度 |
| `clip_dim` | 512 | CLIP ViT-B/32 文本向量维度 |
| `num_patches` | 256 | 图像 patch 数量 (224/14²) |
| `num_heads` | 6 | Transformer 多头注意力头数 |
| `num_layers` | 4 | Transformer 编码器层数 |
| `batch_size` | 32 | 训练批大小 |
| `epochs` | 40 | 训练轮数 |
| `peak_lr` | 5e-4 | ADAMW 峰值学习率 |
| `weight_decay` | 0.2 | 权重衰减 |
| `warmup_epochs` | 4 | 线性预热轮数 |
| `temperature` | 0.07 | InfoNCE 温度系数 |
| `use_audio` | True | 是否启用音频模态 |
| `audio_sample_rate` | 16000 | 音频采样率 |

---

## 📊 损失函数

ConditionNET 使用**自适应加权复合损失**：

$$
\mathcal{L} = \alpha \cdot \mathcal{L}_{condition} + \beta \cdot \mathcal{L}_{consistency}
$$

- **L_condition**：标准 3 类交叉熵损失，负责分类任务
- **L_consistency**：批次内双向 InfoNCE 损失，约束成功 demonstration 中状态变化向量（cls⁺ − cls⁻）与改写动作文本 sp 的一致性
- **α, β 自适应锁定**：首个 batch 后根据两个损失的量级自动平衡，使两者贡献相当

---

## 🎵 音频模态

模型可选地支持音频输入，用于捕获机器人操作时的声音特征（如倾倒、碰撞、电机声等）。

- **编码器**：Mel-spectrogram → 4 层 CNN → Global Pool → Linear
- **集成方式**：音频 token 拼接在 visual CLS 之后，参与 State Transformer 和 Condition Transformer 的 Self-Attention
- **静音处理**：缺失音频时自动填充静音
- **自动回退**：若音频编码器加载失败，训练器自动切换到纯视觉模式

通过 `--no_audio` 或设置 `use_audio=False` 可关闭音频模态。

---

## 🔬 技术栈

| 组件 | 技术 |
|------|------|
| 视觉编码器 | DINOv2-vits14 (冻结) |
| 文本编码器 | CLIP ViT-B/32 (冻结) |
| 音频编码器 | Mel-spectrogram + CNN (可训练) |
| 核心网络 | PyTorch nn.TransformerEncoder |
| 优化器 | ADAMW |
| 学习率调度 | Warmup + Cosine Annealing |
| 损失函数 | CrossEntropy + InfoNCE |

---

## 📝 引用

如果这项研究对你的工作有帮助，欢迎引用：

```bibtex
@misc{conditionnet,
  title   = {ConditionNET: Multi-modal Condition Relationship Prediction for Robot Manipulation},
  author  = {},
  year    = {2025},
}
```

---

## 📄 许可

本项目仅供学术研究使用。
