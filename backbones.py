"""
冻结的特征提取器：DINOv2 (图像) + CLIP (文本)
自动适配 HuggingFace 镜像站以兼容无网络环境。
"""

import os

import torch
import torch.nn as nn
from torchvision import transforms

# AutoDL / 国内环境: 优先用镜像站
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import CLIPTextModel


def _load_dinov2():
    """尝试多种方式加载 DINOv2，优先本地缓存"""
    # 方式1: 从本地缓存加载（跳过 GitHub 检查）
    try:
        model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14",
            source="local", trust_repo=True,
        )
        print("  [DINOv2] 从本地缓存加载成功")
        return model
    except Exception:
        pass

    # 方式2: 从 torch.hub 在线下载
    try:
        model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14",
            trust_repo=True,
        )
        print("  [DINOv2] 从 torch.hub 下载成功")
        return model
    except Exception:
        pass

    # 方式3: 直接下载权重
    try:
        import os
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14",
            trust_repo=True,
        )
        print("  [DINOv2] 从镜像下载成功")
        return model
    except Exception:
        pass

    raise RuntimeError(
        "无法加载 DINOv2 模型。请检查网络或手动下载:\n"
        "  cd /root/.cache/torch/hub/facebookresearch_dinov2_main\n"
        "  git pull  # 更新 repo\n"
    )


class DINOv2Encoder(nn.Module):
    """DINOv2-vits14 图像编码器 (冻结)，提取 patch tokens [B, 256, 384]"""

    def __init__(self):
        super().__init__()
        self.model = _load_dinov2()
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.img_size = 224
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.no_grad()
    def forward(self, images):
        """
        images: PIL Image 列表或 Tensor [B, 3, 224, 224]
        返回: [B, 256, 384] patch tokens
        """
        if not isinstance(images, torch.Tensor):
            images = torch.stack([self.transform(img) for img in images])
        images = images.to(next(self.model.parameters()).device)

        out = self.model.forward_features(images)
        tokens = out["x_norm_patchtokens"]  # [B, 256, 384]
        return tokens


def _load_clip(model_name="openai/clip-vit-base-patch32"):
    """尝试多个源加载 CLIP 模型"""
    from transformers import CLIPTextModel, CLIPTokenizer

    # 先试镜像站，再试官方源
    endpoints = [
        ("HF 镜像站", "https://hf-mirror.com"),
        ("HuggingFace", "https://huggingface.co"),
    ]

    for source_name, endpoint in endpoints:
        try:
            os.environ["HF_ENDPOINT"] = endpoint
            model = CLIPTextModel.from_pretrained(model_name)
            tokenizer = CLIPTokenizer.from_pretrained(model_name)
            print(f"  [CLIP] 从 {source_name} 加载成功")
            return model, tokenizer
        except Exception as e:
            print(f"  [CLIP] {source_name} 加载失败: {e}")
            continue

    raise RuntimeError(
        "无法加载 CLIP 模型。请在能联网的环境预先下载模型到本地，"
        "然后用 CLIPTextEncoder(model_path='你的本地路径') 加载。"
    )


class CLIPTextEncoder(nn.Module):
    """CLIP ViT-B/32 文本编码器 (冻结)，提取语义向量 [B, 512]"""

    def __init__(self, model_path: str | None = None):
        super().__init__()

        if model_path:
            from transformers import CLIPTextModel, CLIPTokenizer
            self.model = CLIPTextModel.from_pretrained(model_path)
            self.tokenizer = CLIPTokenizer.from_pretrained(model_path)
        else:
            self.model, self.tokenizer = _load_clip()

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, texts):
        """
        texts: List[str] 或 str
        返回: [B, 512] CLIP 文本向量
        """
        if isinstance(texts, str):
            texts = [texts]

        device = next(self.model.parameters()).device
        tokens = self.tokenizer(
            texts, padding=True, truncation=True, max_length=77, return_tensors="pt"
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}

        out = self.model(**tokens)
        return out.pooler_output  # [B, 512]
