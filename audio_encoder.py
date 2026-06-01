"""
音频编码器: Mel-spectrogram → CNN → audio token
轻量级可训练模块，用于提取机器人动作声音特征。
"""

import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    """Mel-spectrogram CNN 编码器 → 单个 audio embedding [B, embed_dim]"""

    def __init__(
        self,
        embed_dim: int = 384,
        sample_rate: int = 16000,
        duration: float = 1.0,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 256,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

        import torchaudio  # lazy import

        # Mel-spectrogram 转换器 (不可训练)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=50,
            f_max=sample_rate // 2,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

        # CNN backbone: (1, 128, T) → embed_dim
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.proj = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def compute_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """波形 [B, samples] → Mel谱 [B, 1, n_mels, n_frames]"""
        # 确保长度
        target_samples = int(self.sample_rate * self.duration)
        if waveform.shape[-1] < target_samples:
            waveform = nn.functional.pad(waveform, (0, target_samples - waveform.shape[-1]))
        else:
            waveform = waveform[..., :target_samples]

        mel = self.mel_transform(waveform)      # [B, n_mels, n_frames]
        mel = self.amplitude_to_db(mel)          # [B, n_mels, n_frames]
        mel = (mel + 80) / 80                    # 归一化到 [0, 1]
        return mel.unsqueeze(1)                  # [B, 1, n_mels, n_frames]

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: [B, samples] 原始音频波形
        返回: [B, embed_dim] audio embedding
        """
        mel = self.compute_mel(waveform)          # [B, 1, n_mels, n_frames]
        x = self.backbone(mel)                    # [B, 256, 1, 1]
        x = x.flatten(1)                          # [B, 256]
        return self.proj(x)                       # [B, embed_dim]


class AudioEncoderNoTrain(AudioEncoder):
    """冻结的音频编码器（仅用于提取特征，不参与训练）"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.proj.parameters():
            p.requires_grad = False
        self.eval()
