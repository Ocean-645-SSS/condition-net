"""
ConditionNET 入口脚本
  训练: python main.py --mode train --data_csv data/demos.csv --img_dir data/images
  评估: python main.py --mode eval --data_csv data/demos.csv --img_dir data/images --ckpt checkpoints/best_model.pt
  推理: python main.py --mode infer --image path/to/img.jpg --text "pick the cup"
"""

import argparse
import sys
from pathlib import Path

import torch

from config import Config
from backbones import DINOv2Encoder, CLIPTextEncoder
from condition_net import ConditionNET
from dataset import build_loaders
from trainer import Trainer


# ═══════════════════════════════════════════════════════
# 训练
# ═══════════════════════════════════════════════════════

def run_train(args):
    cfg_kwargs = dict(
        batch_size=args.batch_size,
        epochs=args.epochs,
        data_root=args.data_csv,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
    )
    if args.device is not None:
        cfg_kwargs["device"] = args.device
    cfg = Config(**cfg_kwargs)

    # ImperfectPour 模式: 使用预定义的 train/val split
    if args.dataset == "imperfectpour":
        from dataset import build_loaders_from_splits

        train_csv = args.train_csv or cfg.imperfectpour_train_csv
        val_csv = args.val_csv or cfg.imperfectpour_val_csv
        img_dir = args.img_dir or cfg.imperfectpour_img_dir

        train_loader, val_loader, cons_loader = build_loaders_from_splits(
            train_csv=train_csv,
            val_csv=val_csv,
            img_dir=img_dir,
            paraphrase_file=args.paraphrase_file,
            batch_size=cfg.batch_size,
            num_workers=args.num_workers,
        )
    else:
        train_loader, val_loader, cons_loader = build_loaders(
            csv_path=args.data_csv,
            img_dir=args.img_dir,
            paraphrase_file=args.paraphrase_file,
            batch_size=cfg.batch_size,
            train_ratio=cfg.train_val_split,
            num_workers=args.num_workers,
        )

    trainer = Trainer(cfg)
    if args.ckpt:
        trainer.load(args.ckpt)

    trainer.fit(train_loader, val_loader, cons_loader)


# ═══════════════════════════════════════════════════════
# 评估
# ═══════════════════════════════════════════════════════

@torch.no_grad()
def run_eval(args):
    cfg = Config()
    device = torch.device(args.device or cfg.device)

    # 加载模型
    dino = DINOv2Encoder().to(device)
    clip_enc = CLIPTextEncoder().to(device)
    model = ConditionNET().to(device)
    model.eval()

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[Eval] 加载模型: {args.ckpt} (epoch {ckpt['epoch']+1})")

    # 加载数据
    _, val_loader, _ = build_loaders(
        csv_path=args.data_csv,
        img_dir=args.img_dir,
        batch_size=cfg.batch_size,
        train_ratio=cfg.train_val_split,
    )

    correct = 0
    total = 0
    class_correct = [0, 0, 0]
    class_total = [0, 0, 0]
    class_names = ["precondition", "effect", "unsatisfied"]

    for batch in val_loader:
        images = batch["images"]
        texts = batch["action_texts"]
        labels = batch["labels"].to(device)

        vt = dino(images).to(device)
        te = clip_enc(texts).to(device)
        logits, E, state_cls = model(vt, te)

        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += len(labels)

        for i, label in enumerate(labels):
            class_total[label] += 1
            if preds[i] == label:
                class_correct[label] += 1

    print(f"\n[Eval] 总准确率: {correct/total:.2%} ({correct}/{total})")
    for i, name in enumerate(class_names):
        if class_total[i] > 0:
            print(f"  {name}: {class_correct[i]/class_total[i]:.2%} ({class_correct[i]}/{class_total[i]})")


# ═══════════════════════════════════════════════════════
# 推理 (单张图像 + 单条指令)
# ═══════════════════════════════════════════════════════

@torch.no_grad()
def run_infer(args):
    from PIL import Image

    cfg = Config()
    device = torch.device(args.device or cfg.device)
    class_names = ["precondition", "effect", "unsatisfied"]

    dino = DINOv2Encoder().to(device)
    clip_enc = CLIPTextEncoder().to(device)
    model = ConditionNET().to(device)
    model.eval()

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    image = Image.open(args.image).convert("RGB")
    vt = dino([image]).to(device)
    te = clip_enc([args.text]).to(device)
    logits, E, state_cls = model(vt, te)

    probs = torch.softmax(logits, dim=-1)[0]
    pred = logits.argmax(dim=-1).item()

    print(f"\n[Infer] 图像: {args.image}")
    print(f"[Infer] 动作: {args.text}")
    print(f"[Infer] 预测: {class_names[pred]} (置信度 {probs[pred]:.2%})")
    for i, name in enumerate(class_names):
        print(f"  {name}: {probs[i]:.4f}")


# ═══════════════════════════════════════════════════════
# 参数解析
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ConditionNET")
    parser.add_argument("--mode", choices=["train", "eval", "infer"], default="train")
    parser.add_argument("--dataset", choices=["default", "imperfectpour"], default="imperfectpour",
                        help="数据集类型")

    # 数据
    parser.add_argument("--data_csv", type=str, help="CSV 文件路径 (default 模式)")
    parser.add_argument("--train_csv", type=str, default=None, help="训练集 CSV (imperfectpour 模式)")
    parser.add_argument("--val_csv", type=str, default=None, help="验证集 CSV (imperfectpour 模式)")
    parser.add_argument("--img_dir", type=str, help="图像根目录")
    parser.add_argument("--paraphrase_file", type=str, default=None, help="改写文本 JSON")

    # 训练
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)

    # 保存/恢复
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--ckpt", type=str, default=None, help="模型检查点路径")

    # 推理
    parser.add_argument("--image", type=str, help="输入图像 (infer 模式)")
    parser.add_argument("--text", type=str, help="动作描述 (infer 模式)")

    args = parser.parse_args()

    if args.mode == "train":
        if args.dataset == "default" and (not args.data_csv or not args.img_dir):
            print("错误: default 模式需要 --data_csv 和 --img_dir")
            sys.exit(1)
        run_train(args)
    elif args.mode == "eval":
        if not args.ckpt or not args.data_csv or not args.img_dir:
            print("错误: --ckpt, --data_csv 和 --img_dir 是评估模式的必选参数")
            sys.exit(1)
        run_eval(args)
    elif args.mode == "infer":
        if not args.ckpt or not args.image or not args.text:
            print("错误: --ckpt, --image 和 --text 是推理模式的必选参数")
            sys.exit(1)
        run_infer(args)


if __name__ == "__main__":
    main()
