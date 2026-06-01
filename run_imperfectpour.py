"""
ImperfectPour 一键训练脚本
  1. 检查/执行数据预处理（从视频提取帧并生成 CSV）
  2. 开始 ConditionNET 训练

用法:
  python run_imperfectpour.py
  python run_imperfectpour.py --dataset_root /root/autodl-tmp/ImperfectPour/ImperfectPour
  python run_imperfectpour.py --epochs 40 --batch_size 16 --skip_preprocess
"""

import argparse
import subprocess
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
DATA_DIR = SRC_DIR / "data" / "imperfectpour"


def check_preprocessed() -> bool:
    train_csv = DATA_DIR / "train.csv"
    val_csv = DATA_DIR / "val.csv"
    frames_dir = DATA_DIR / "frames"
    return train_csv.exists() and val_csv.exists() and frames_dir.exists()


def run_preprocess(dataset_root: str | None = None, skip_audio: bool = False):
    print("\n>>> 预处理数据不存在，开始提取视频帧...\n")
    cmd = [sys.executable, str(SRC_DIR / "preprocess_imperfectpour.py")]
    if dataset_root:
        cmd += ["--dataset_root", dataset_root]
    if skip_audio:
        cmd += ["--skip_audio"]
    result = subprocess.run(cmd, cwd=str(SRC_DIR))
    if result.returncode != 0:
        print("\n✗ 预处理失败! 请检查 ffmpeg 是否已安装，数据集路径是否正确。")
        print("  安装 ffmpeg: sudo apt install ffmpeg -y")
        sys.exit(1)


def run_training(args):
    cmd = [
        sys.executable, str(SRC_DIR / "main.py"),
        "--mode", "train",
        "--dataset", "imperfectpour",
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--output_dir", str(args.output_dir),
        "--log_dir", str(args.log_dir),
    ]
    if args.num_workers:
        cmd += ["--num_workers", str(args.num_workers)]
    if args.device:
        cmd += ["--device", args.device]

    print(f"\n>>> 开始训练...\n")
    subprocess.run(cmd, cwd=str(SRC_DIR))


def main():
    parser = argparse.ArgumentParser(description="ImperfectPour + ConditionNET 一键训练")
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="ImperfectPour 数据集根目录")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--skip_preprocess", action="store_true",
                        help="跳过预处理")
    parser.add_argument("--skip_audio", action="store_true",
                        help="不提取音频（预处理时跳过音频）")
    parser.add_argument("--no_audio", action="store_true",
                        help="训练时不使用音频模态")
    args = parser.parse_args()

    print("=" * 60)
    print("ImperfectPour → ConditionNET 训练")
    print("=" * 60)

    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:    {torch.cuda.is_available()}")

    if not args.skip_preprocess and not check_preprocessed():
        run_preprocess(args.dataset_root, skip_audio=args.skip_audio)
    else:
        print("\n>>> 数据已预处理，跳过提取步骤")
        if not args.skip_audio:
            print("  (提示: 如需补充提取音频，请删除 data/imperfectpour/train.csv 后重新运行)")

    if args.no_audio:
        print(">>> 音频模态已关闭")
        # 通过修改 config 默认值来禁用音频
        from config import Config
        Config.use_audio = False

    run_training(args)


if __name__ == "__main__":
    main()
