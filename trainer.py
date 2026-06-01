"""
ConditionNET 训练器
  ADAMW + cosine annealing + warmup
  复合损失: L = α·L_condition + β·L_consistency
  支持视觉+音频+文本三模态
"""

import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from backbones import DINOv2Encoder, CLIPTextEncoder
from condition_net import ConditionNET
from losses import ConditionLoss
from config import Config


class Trainer:
    def __init__(self, config: Config):
        self.cfg = config
        self.device = torch.device(config.device)
        self.global_step = 0
        self.current_epoch = 0

        # 冻结的特征提取器
        self.dino = DINOv2Encoder().to(self.device)
        self.clip = CLIPTextEncoder().to(self.device)

        # 可训练模型
        self.model = ConditionNET(
            embed_dim=config.embed_dim,
            clip_dim=config.clip_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
        ).to(self.device)

        # 音频编码器 (可训练)
        self.use_audio = config.use_audio
        self.audio_encoder = None
        if self.use_audio:
            try:
                from audio_encoder import AudioEncoder
                self.audio_encoder = AudioEncoder(
                    embed_dim=config.embed_dim,
                    sample_rate=config.audio_sample_rate,
                    duration=config.audio_duration,
                    n_mels=config.audio_n_mels,
                ).to(self.device)
            except (ImportError, OSError, RuntimeError) as e:
                print(f"[Trainer] ⚠ 音频编码器加载失败: {e}")
                print("[Trainer] → 自动回退到纯视觉模式，不影响训练")
                self.use_audio = False
                self.audio_encoder = None

        # 损失函数
        self.criterion = ConditionLoss(temperature=config.temperature).to(self.device)

        # 优化器 (ConditionNET + criterion 投影层 + audio_encoder)
        trainable = list(self.model.parameters()) + \
                    list(self.criterion.text_proj_consistency.parameters())
        if self.audio_encoder is not None:
            trainable += list(self.audio_encoder.parameters())
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=config.peak_lr,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
        )

        self.scheduler = None

        # 日志
        self.log_path = Path(config.log_dir) / "train_log.csv"
        self._init_log()

        # 最佳模型追踪
        self.best_val_acc = 0.0
        self.best_path = Path(config.output_dir) / "best_model.pt"

        print(f"[Trainer] 设备: {self.device}")
        print(f"[Trainer] 音频模态: {'启用' if self.use_audio else '关闭'}")
        trainable_count = sum(p.numel() for p in trainable)
        print(f"[Trainer] 可训练参数: {trainable_count:,}")

    def _init_log(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("epoch,step,train_loss,l_cond,l_cons,val_loss,val_acc,lr,time\n")

    # ═══════════════════════════════════════════════════════════
    # 特征提取
    # ═══════════════════════════════════════════════════════════

    def _extract_visual(self, images):
        # 处理字符串路径（consistency batch）和 PIL 图像（分类 batch）
        from PIL import Image as PILImage
        imgs = []
        for img in images:
            if isinstance(img, str):
                imgs.append(PILImage.open(img).convert("RGB"))
            else:
                imgs.append(img)
        return self.dino(imgs).to(self.device)

    def _extract_text(self, texts):
        return self.clip(texts).to(self.device)

    def _extract_audio(self, audio_paths):
        """加载 .wav 文件 → 音频波形 → AudioEncoder → [B, 1, embed_dim]"""
        if not self.use_audio or self.audio_encoder is None:
            return None

        import torchaudio  # lazy import，避免 CUDA 版本不匹配

        waveforms = []
        for path in audio_paths:
            if path and Path(path).exists():
                wav, sr = torchaudio.load(path)
                if sr != self.cfg.audio_sample_rate:
                    wav = torchaudio.functional.resample(
                        wav, sr, self.cfg.audio_sample_rate)
                # 单声道
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                waveforms.append(wav.squeeze(0))  # [samples]
            else:
                # 静音填充
                samples = int(self.cfg.audio_sample_rate * self.cfg.audio_duration)
                waveforms.append(torch.zeros(samples))

        if not waveforms:
            return None

        # Pad 到相同长度
        max_len = max(w.shape[-1] for w in waveforms)
        padded = []
        for w in waveforms:
            if w.shape[-1] < max_len:
                w = nn.functional.pad(w, (0, max_len - w.shape[-1]))
            padded.append(w)
        wav_batch = torch.stack(padded).to(self.device)  # [B, samples]

        audio_emb = self.audio_encoder(wav_batch)          # [B, embed_dim]
        return audio_emb.unsqueeze(1)                       # [B, 1, embed_dim]

    # ═══════════════════════════════════════════════════════════
    # 单步训练
    # ═══════════════════════════════════════════════════════════

    def train_step(self, class_batch, cons_batch):
        """执行一步训练，返回 (total_loss, l_cond, l_cons)"""

        # ── 分类前向 ──
        images = class_batch["images"]
        texts = class_batch["action_texts"]
        labels = class_batch["labels"].to(self.device)
        audio_paths = class_batch.get("audio_paths", [])

        vt_class = self._extract_visual(images)
        te_class = self._extract_text(texts)
        audio_token = self._extract_audio(audio_paths) if audio_paths else None
        logits, E, state_cls = self.model(vt_class, te_class, audio_token=audio_token)

        # ── 一致性前向 ──
        vt_pre = self._extract_visual(cons_batch["img_pre"])
        vt_post = self._extract_visual(cons_batch["img_post"])
        text_sp = self._extract_text(cons_batch["paraphrases"])

        audio_pre = self._extract_audio(cons_batch.get("audio_pre_paths", []))
        audio_post = self._extract_audio(cons_batch.get("audio_post_paths", []))

        state_pre = self.model.state_transformer(vt_pre, audio_token=audio_pre)
        state_post = self.model.state_transformer(vt_post, audio_token=audio_post)
        cls_pre = state_pre[:, 0, :]
        cls_post = state_post[:, 0, :]

        success_mask = torch.ones(
            len(cons_batch["paraphrases"]), dtype=torch.bool, device=self.device
        )

        # ── 总损失 ──
        total_loss, l_cond, l_cons = self.criterion(
            logits, labels, cls_pre, cls_post, text_sp, success_mask,
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), l_cond, l_cons

    # ═══════════════════════════════════════════════════════════
    # 验证
    # ═══════════════════════════════════════════════════════════

    @torch.no_grad()
    def validate(self, val_loader: DataLoader):
        self.model.eval()
        if self.audio_encoder:
            self.audio_encoder.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch in val_loader:
            images = batch["images"]
            texts = batch["action_texts"]
            labels = batch["labels"].to(self.device)
            audio_paths = batch.get("audio_paths", [])

            vt = self._extract_visual(images)
            te = self._extract_text(texts)
            audio_token = self._extract_audio(audio_paths) if audio_paths else None
            logits, E, state_cls = self.model(vt, te, audio_token=audio_token)

            l_cond = nn.functional.cross_entropy(logits, labels)
            total_loss += l_cond.item() * len(labels)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += len(labels)

        self.model.train()
        if self.audio_encoder:
            self.audio_encoder.train()
        return total_loss / total, correct / total if total > 0 else 0.0

    # ═══════════════════════════════════════════════════════════
    # 学习率调度
    # ═══════════════════════════════════════════════════════════

    def _get_lr(self, step: int, total_steps: int):
        warmup_steps = self.cfg.warmup_epochs * (total_steps // self.cfg.epochs)
        warmup_steps = max(warmup_steps, 1)
        if step < warmup_steps:
            return self.cfg.peak_lr * step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return self.cfg.peak_lr * 0.5 * (1 + math.cos(math.pi * progress))

    def _set_lr(self, step: int, total_steps: int):
        lr = self._get_lr(step, total_steps)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    # ═══════════════════════════════════════════════════════════
    # 主训练循环
    # ═══════════════════════════════════════════════════════════

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        consistency_loader: DataLoader | None = None,
    ):
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * self.cfg.epochs
        self.model.train()
        if self.audio_encoder:
            self.audio_encoder.train()

        print(f"[Trainer] 开始训练 — {self.cfg.epochs} epochs, {steps_per_epoch} steps/epoch")

        for epoch in range(self.current_epoch, self.cfg.epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0
            epoch_l_cond = 0.0
            epoch_l_cons = 0.0
            t0 = time.time()

            cons_iter = iter(consistency_loader) if consistency_loader else None

            for step, class_batch in enumerate(train_loader):
                if cons_iter is not None:
                    try:
                        cons_batch = next(cons_iter)
                    except StopIteration:
                        cons_iter = iter(consistency_loader)
                        cons_batch = next(cons_iter)
                else:
                    cons_batch = self._make_cons_from_class(class_batch)
                    if cons_batch is None:
                        continue

                gs = epoch * steps_per_epoch + step
                lr = self._set_lr(gs, total_steps)

                loss, l_cond, l_cons = self.train_step(class_batch, cons_batch)
                self.global_step += 1

                epoch_loss += loss
                epoch_l_cond += l_cond
                epoch_l_cons += l_cons

                if (step + 1) % 50 == 0:
                    avg_l = epoch_loss / (step + 1)
                    print(f"  Epoch {epoch+1}/{self.cfg.epochs} | "
                          f"Step {step+1}/{steps_per_epoch} | "
                          f"Loss {avg_l:.4f} | LR {lr:.2e}")

            n_steps = steps_per_epoch
            avg_loss = epoch_loss / n_steps
            avg_cond = epoch_l_cond / n_steps
            avg_cons = epoch_l_cons / n_steps

            val_loss, val_acc = self.validate(val_loader)
            elapsed = time.time() - t0

            print(f"── Epoch {epoch+1} ── "
                  f"Train Loss: {avg_loss:.4f} "
                  f"(c: {avg_cond:.4f}, s: {avg_cons:.4f}) | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.2%} | "
                  f"Time: {elapsed:.0f}s")

            self._log(epoch, avg_loss, avg_cond, avg_cons, val_loss, val_acc, lr, elapsed)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save(self.best_path)
                print(f"  ✓ 保存最佳模型 (acc={val_acc:.2%})")

        print(f"[Trainer] 训练完成! 最佳 val acc: {self.best_val_acc:.2%}")

    def _make_cons_from_class(self, class_batch):
        pair_ids = class_batch["demo_ids"]
        is_pre = class_batch["is_pre"]
        has_audio = "audio_paths" in class_batch

        img_pre, img_post, paraphrases = [], [], []
        audio_pre_paths, audio_post_paths = [], []

        pre_indices = {}
        for i in range(len(pair_ids)):
            pid = pair_ids[i].item()
            if pid < 0:
                continue
            if is_pre[i]:
                pre_indices[pid] = i
            elif pid in pre_indices:
                pi = pre_indices[pid]
                img_pre.append(class_batch["images"][pi])
                img_post.append(class_batch["images"][i])
                paraphrases.append(class_batch["paraphrases"][i])
                if has_audio:
                    audio_pre_paths.append(class_batch["audio_paths"][pi])
                    audio_post_paths.append(class_batch["audio_paths"][i])
                del pre_indices[pid]

        if len(img_pre) < 2:
            return None

        result = {
            "img_pre": img_pre,
            "img_post": img_post,
            "paraphrases": paraphrases,
        }
        if has_audio:
            result["audio_pre_paths"] = audio_pre_paths
            result["audio_post_paths"] = audio_post_paths
        return result

    # ═══════════════════════════════════════════════════════════
    # 日志 & 保存
    # ═══════════════════════════════════════════════════════════

    def _log(self, epoch, loss, l_cond, l_cons, val_loss, val_acc, lr, elapsed):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch+1},{self.global_step},{loss:.6f},{l_cond:.6f},"
                    f"{l_cons:.6f},{val_loss:.6f},{val_acc:.6f},{lr:.2e},{elapsed:.0f}\n")

    def save(self, path):
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "criterion_state_dict": self.criterion.state_dict(),
            "best_val_acc": self.best_val_acc,
        }
        if self.audio_encoder is not None:
            checkpoint["audio_encoder_state_dict"] = self.audio_encoder.state_dict()
        torch.save(checkpoint, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.criterion.load_state_dict(ckpt["criterion_state_dict"])
        self.current_epoch = ckpt["epoch"] + 1
        self.global_step = ckpt["global_step"]
        self.best_val_acc = ckpt["best_val_acc"]
        if self.audio_encoder is not None and "audio_encoder_state_dict" in ckpt:
            self.audio_encoder.load_state_dict(ckpt["audio_encoder_state_dict"])
        print(f"[Trainer] 从 {path} 恢复 (epoch {ckpt['epoch']+1}, acc={ckpt['best_val_acc']:.2%})")
