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
│   └── requirements.txt             # Python 依赖
├── main.py                          # 早期原型脚本
├── condition-net.pdf                # 论文/文档
└── ImperfectPour/                   # 数据集（需自行解压）
```

---

## 📦 ImperfectPour 数据集

ConditionNET 使用 **ImperfectPour** 数据集进行训练和评估。这是一个 **Franka Emika Panda 机器人执行酒吧调酒任务** 的 VR 遥操作演示数据集，专注于收集失败演示以支持异常检测和执行监控研究。

### 🎯 任务概述

机器人需要完成倒饮料的完整流程，在发生泼洒时需要清理桌面。每个演示录制包含 **双视角视频** + **帧级本体感知数据** + **动作标注**。

### 📊 数据规模总览

| 指标 | 数值 |
|------|------|
| 唯一录制片段 | 118 个 |
| 相机视角 | 2 个（cam2 + cam3），共 236 个标注 |
| 动作总条数 | 1,096 条 |
| 帧级 NPZ 数据 | 296,319 个 |
| 视频文件 | 282 个 MP4（含 46 个无标注的额外视频） |

### ✅ 成功与失败演示

| 类型 | 数量 | 占比 |
|------|------|------|
| **成功演示** 🟢 | 78 | 33.1% |
| **失败演示** 🔴 | 158 | 66.9% |

> 💡 这是一个**以失败为主**的数据集（失败率 ~67%），非常契合 ConditionNET 的条件关系推理任务 —— 模型需要判断一个动作在给定场景下是否"满足条件"。

**失败原因分布**（一个演示可同时包含多种失败）：

| 失败标签 | 说明 | 出现次数 |
|----------|------|----------|
| `has_spill` | 液体泼洒 | 356 次 |
| `missing_object` | 物体缺失（如没有瓶子可拿） | 176 次 |
| `has_fallen` | 物体掉落 | 8 次 |

### 🏷️ 6 种动作类型

机器人按以下流水线执行任务，每个动作有明确的前置条件（precondition）和效果（effect）帧范围：

| 序号 | 动作 | 英文标识 | 数量 | 占比 |
|------|------|------|------|------|
| 1 | 拿起瓶子 | `pick up bottle` | 252 | 23.0% |
| 2 | 倒果汁入杯 | `pour juice into cup` | 226 | 20.6% |
| 3 | 放置瓶子 | `place bottle on table` | 244 | 22.3% |
| 4 | 拿起抹布 | `pick up cloth` | 148 | 13.5% |
| 5 | 擦拭桌子 | `wipe table` | 94 | 8.6% |
| 6 | 放置抹布 | `place cloth on table` | 132 | 12.0% |

**完整任务流程**：拿瓶子 → 倒果汁 → 放瓶子 → 拿抹布 → 擦桌子 → 放抹布

每个标注文件包含 2-14 个动作，其中 3 或 6 个动作最为常见（对应完整调酒流程）。

### 📁 数据目录结构

每个录制目录包含一个视频文件和逐帧本体感知数据：

```
ImperfectPour/
├── annotations/                         # 动作标注 (236 个 JSON)
│   ├── recording_2023-11-21-16-41-33_cam2.json
│   ├── recording_2023-11-21-16-41-33_cam3.json
│   └── ...
├── videos/                              # 视频 + 本体感知数据 (282 个目录)
│   └── recording_2023-11-21-16-41-33_cam2/
│       ├── recording_..._cam2.mp4       # 视频文件 (30 FPS)
│       ├── 00000.npz                    # 第 0 帧本体感知数据
│       ├── 00001.npz                    # 第 1 帧
│       └── ...                          # 约 800-2000 帧
├── train_split.txt                      # 训练集划分 (94 录制 / 188 标注)
├── val_split.txt                        # 验证集划分 (24 录制 / 48 标注)
├── extract_frames.py                    # 视频帧提取脚本
├── create_vids.py                       # 帧→视频还原脚本
└── README.md                            # 数据集原始说明
```

### 🤖 本体感知数据（.npz 文件）

每帧 `.npz` 文件包含 **56 个字段**，覆盖机器人的完整运动学和力学信息：

| 类别 | 字段 | 说明 |
|------|------|------|
| **夹爪** | `finger_0_pos`, `finger_1_pos`, `finger_0_vel`, `finger_1_vel` | 夹爪位置与速度 |
| **关节位置 (7轴)** | `pos_joint1` ~ `pos_joint7` | 各关节角度位置 |
| **关节速度 (7轴)** | `vel_joint1` ~ `vel_joint7` | 各关节角速度 |
| **关节力矩 (7轴)** | `eff_joint1` ~ `eff_joint7` | 各关节施加力矩 |
| **末端位姿** | `pose_x/y/z`, `pose_qx/qy/qz/qw` | 末端执行器位置（3D）+ 四元数旋转 |
| **末端速度** | `vel_x/y/z`, `vel_qx/qy/qz` | 末端线速度 + 角速度 |
| **估计力/力矩** | `est_force_x/y/z`, `est_torque_x/y/z` | AIDIN AFT200 传感器估计值 |
| **测量力/力矩** | `measured_force_x/y/z`, `measured_torque_x/y/z` | 传感器直接测量值 |
| **法兰力/力矩** | `flanch_force_x/y/z`, `flanch_torque_x/y/z` | 法兰坐标系下的力/力矩 |

读取示例：
```python
>>> import numpy as np
>>> data = np.load("00951.npz", allow_pickle=True)["arr_0"][()]
>>> data.keys()
dict_keys(['finger_0_pos', 'finger_1_pos', 'finger_0_vel', 'finger_1_vel',
           'est_force_x', 'est_force_y', 'est_force_z', ...])
```

### 📝 标注格式详解

每个 JSON 文件对应一个录制的全部动作标注：

```json
[
    {
        "action": "pour juice into cup",
        "pre_start_fid": 320,
        "pre_end_fid": 370,
        "post_start_fid": 642,
        "post_end_fid": 723,
        "has_spill": false,
        "missing_object": false,
        "do_augmentation": true,
        "has_fallen": false
    }
]
```

**字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `action` | string | 动作描述文本（用于 ConditionNET 的文本编码） |
| `pre_start_fid` / `pre_end_fid` | int | 前置条件帧范围（动作执行前的状态） |
| `post_start_fid` / `post_end_fid` | int | 效果帧范围（动作执行后的结果） |
| `has_spill` | bool | 是否发生液体泼洒 |
| `missing_object` | bool | 是否有物体缺失 |
| `has_fallen` | bool | 是否有物体掉落（部分文件有此字段） |
| `do_augmentation` | bool | 该动作是否参与数据增强（True 占 70.1%，False 占 29.9%） |

> ⚠️ 当 `missing_object=true` 时，帧范围通常设为 `[0, 全视频长度]` 作为占位符，表示该物体未出现在场景中。这类样本的 `do_augmentation` 一律为 `false`。

### 🔄 与 ConditionNET 的数据流

`preprocess_imperfectpour.py` 脚本将原始数据集转换为模型可用的格式：

```
原始数据                          ConditionNET 输入
─────────────────────────────────────────────────
pre_end_fid 帧     ──→   I⁻ (前置条件图像)
post_start_fid 帧  ──→   I⁺ (效果图像)
对应帧音频片段     ──→   音频 token (可选)
action 字段文本    ──→   动作指令文本

三类标签对应：
  I⁻ + 动作文本   →  Precondition (0)
  I⁺ + 动作文本   →  Effect (1)
  跨 demo 随机配对  →  Unsatisfied (2)
```

### 📅 数据采集信息

- **采集时间**：2023年11月 ~ 2024年1月，分布在多个日期
- **硬件**：AIDIN AFT200 系列力/力矩传感器
- **力传感器品牌**：AIDIN AFT200 Series

### 🗂️ 训练/验证划分

| 划分 | 录制数 | 标注文件 | 对应文件 |
|------|--------|----------|----------|
| **训练集** | 94 | 188（94 × 2 视角） | `train_split.txt` |
| **验证集** | 24 | 48（24 × 2 视角） | `val_split.txt` |

两个视角（cam2、cam3）的标注内容**完全一致**（帧编号、动作、标签均相同），各自对应不同视角的视频数据。

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
