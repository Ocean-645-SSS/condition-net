#!/bin/bash
# ============================================================
# AutoDL 环境一键配置脚本
# 用法: bash setup_autodl.sh
# ============================================================
set -e

echo "=============================================="
echo " ConditionNET + ImperfectPour AutoDL 配置"
echo "=============================================="

# ── 1. 安装系统依赖 (ffmpeg) ──
echo ""
echo "[1/4] 安装 ffmpeg..."
if ! command -v ffmpeg &> /dev/null; then
    sudo apt update -qq && sudo apt install ffmpeg -y -qq
    echo "  ✓ ffmpeg 安装完成"
else
    echo "  ✓ ffmpeg 已存在"
fi

# ── 2. 安装 Python 依赖 ──
echo ""
echo "[2/4] 安装 Python 依赖..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 2>/dev/null || \
pip install torch torchvision
pip install transformers pillow tqdm numpy

echo "  ✓ Python 依赖安装完成"

# ── 3. 解压数据集 ──
echo ""
echo "[3/4] 查找并解压 ImperfectPour 数据集..."

# AutoDL 常见路径
ZIP_PATHS=(
    "/root/autodl-tmp/ImperfectPour.zip"
    "/root/ImperfectPour.zip"
    "/root/autodl-tmp/ImperfectPour"
    "/root/autodl-tmp/ImperfectPour/ImperfectPour"
)

DATASET_DIR=""
for path in "${ZIP_PATHS[@]}"; do
    if [[ "$path" == *.zip ]] && [ -f "$path" ]; then
        ZIP_DIR=$(dirname "$path")
        echo "  发现压缩包: $path"
        echo "  正在解压到: $ZIP_DIR ..."
        if ! command -v unzip &> /dev/null; then
            sudo apt install unzip -y -qq
        fi
        unzip -o "$path" -d "$ZIP_DIR" | tail -1
        echo "  ✓ 解压完成"
        DATASET_DIR="$ZIP_DIR/ImperfectPour/ImperfectPour"
        break
    elif [ -d "$path" ] && [ -d "$path/videos" ]; then
        DATASET_DIR="$path"
        echo "  找到已解压数据集: $DATASET_DIR"
        break
    fi
done

if [ -z "$DATASET_DIR" ]; then
    echo "  ⚠ 未找到数据集，请手动指定:"
    echo "    python preprocess_imperfectpour.py --dataset_root <你的路径>"
else
    echo "  ✓ 数据集目录: $DATASET_DIR"
fi

# ── 4. 验证环境 ──
echo ""
echo "[4/4] 验证环境..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA:    {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:     {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:    {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

echo ""
echo "=============================================="
echo " ✓ 配置完成!"
echo ""
if [ -n "$DATASET_DIR" ]; then
    echo "  运行训练:"
    echo "    python run_imperfectpour.py --dataset_root $DATASET_DIR"
else
    echo "  运行训练 (请替换 <DATA_DIR>):"
    echo "    python run_imperfectpour.py --dataset_root <DATA_DIR>"
fi
echo "=============================================="
