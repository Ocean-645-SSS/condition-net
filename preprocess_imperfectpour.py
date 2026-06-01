"""
ImperfectPour 数据集预处理
从视频中提取动作前后帧 + 音频片段，生成 training CSV。

用法:
  python preprocess_imperfectpour.py
  python preprocess_imperfectpour.py --dataset_root /root/autodl-tmp/ImperfectPour/ImperfectPour
  python preprocess_imperfectpour.py --dataset_root ./ImperfectPour/ImperfectPour --output_dir ./data/imperfectpour
  python preprocess_imperfectpour.py ... --skip_audio    # 仅提取图像，跳过音频
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


AUDIO_DURATION = 1.0    # 音频片段长度 (秒)
AUDIO_SR = 16000         # 采样率
VIDEO_FPS = 30           # 视频帧率


def get_default_dataset_root() -> Path:
    candidates = [
        Path("/root/autodl-tmp/ImperfectPour/ImperfectPour"),
        Path("/root/autodl-tmp/ImperfectPour"),
        Path(__file__).resolve().parents[2] / "ImperfectPour" / "ImperfectPour",
        Path("ImperfectPour/ImperfectPour"),
    ]
    for c in candidates:
        if (c / "videos").exists() and (c / "annotations").exists():
            return c
    return candidates[0]


def read_split(split_file: Path) -> set[str]:
    names = set()
    with open(split_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                names.add(line.replace(".json", ""))
    return names


def extract_frame(video_path: Path, frame_id: int, output_path: Path, fps: int = VIDEO_FPS) -> bool:
    """用 ffmpeg 提取视频的指定帧"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return True

    ts = frame_id / fps
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{ts:.2f}",
        "-i", str(video_path),
        "-vframes", "1",
        "-q:v", "2",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=30)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def extract_audio(
    video_path: Path,
    frame_id: int,
    output_path: Path,
    fps: int = VIDEO_FPS,
    duration: float = AUDIO_DURATION,
    sample_rate: int = AUDIO_SR,
) -> bool:
    """
    从视频中提取以目标帧为中心的一小段音频。
    output_path: .wav 文件路径
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return True

    half_dur = duration / 2.0
    center_ts = frame_id / fps
    start_ts = max(0, center_ts - half_dur)

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_ts:.3f}",
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-vn",                         # 不要视频流
        "-ac", "1",                     # 单声道
        "-ar", str(sample_rate),        # 采样率
        "-sample_fmt", "s16",           # 16-bit
        str(output_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=30)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def process_recordings(
    split_file: Path,
    csv_path: Path,
    videos_dir: Path,
    annotations_dir: Path,
    frames_dir: Path,
    split_name: str,
    skip_audio: bool = False,
):
    recording_names = sorted(read_split(split_file))
    total_names = len(recording_names)
    total_actions = 0
    missing = 0
    audio_failures = 0

    print(f"[{split_name}] 共 {total_names} 个 recording，开始处理...")

    with open(csv_path, "w", encoding="utf-8") as csv_f:
        header = "pre_image_path,post_image_path,pre_audio_path,post_audio_path,action_text\n"
        csv_f.write(header)

        for idx, name in enumerate(recording_names):
            video_dir = videos_dir / name
            annotation_file = annotations_dir / f"{name}.json"
            video_file = video_dir / f"{name}.mp4"

            if not annotation_file.exists():
                print(f"  [{idx+1}/{total_names}] SKIP 缺少标注: {name}")
                missing += 1
                continue
            if not video_file.exists():
                print(f"  [{idx+1}/{total_names}] SKIP 缺少视频: {name}")
                missing += 1
                continue

            with open(annotation_file, "r") as f:
                actions = json.load(f)

            for act in actions:
                pre_fid = act["pre_end_fid"]
                post_fid = act["post_start_fid"]
                action_text = act["action"]

                # 图像路径
                pre_img_rel = f"{name}/pre_{pre_fid:05d}.png"
                post_img_rel = f"{name}/post_{post_fid:05d}.png"
                pre_img_abs = frames_dir / pre_img_rel
                post_img_abs = frames_dir / post_img_rel

                # 音频路径
                pre_audio_rel = f"{name}/audio_{pre_fid:05d}.wav"
                post_audio_rel = f"{name}/audio_{post_fid:05d}.wav"
                pre_audio_abs = frames_dir / pre_audio_rel
                post_audio_abs = frames_dir / post_audio_rel

                # 提取帧
                if not extract_frame(video_file, pre_fid, pre_img_abs):
                    print(f"  [WARN] 帧提取失败: {name} frame {pre_fid}")
                    continue
                if not extract_frame(video_file, post_fid, post_img_abs):
                    print(f"  [WARN] 帧提取失败: {name} frame {post_fid}")
                    continue

                # 提取音频
                if not skip_audio:
                    if not extract_audio(video_file, pre_fid, pre_audio_abs):
                        audio_failures += 1
                    if not extract_audio(video_file, post_fid, post_audio_abs):
                        audio_failures += 1

                csv_f.write(
                    f"{pre_img_rel},{post_img_rel},{pre_audio_rel},{post_audio_rel},{action_text}\n"
                )
                total_actions += 1

            audio_info = "" if skip_audio else f" | {audio_failures} audio fails"
            print(f"  [{idx+1}/{total_names}] ✓ {name} ({len(actions)} actions{audio_info})",
                  flush=True)

    audio_note = " (无音频)" if skip_audio else ""
    print(f"[{split_name}] 写入 {total_actions} 条记录 → {csv_path}{audio_note}")
    if missing:
        print(f"[{split_name}] {missing} 个 recording 缺少数据")


def main():
    parser = argparse.ArgumentParser(description="ImperfectPour 数据预处理")
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="数据集根目录 (包含 videos/, annotations/, *_split.txt)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录 (默认: src/data/imperfectpour)")
    parser.add_argument("--skip_audio", action="store_true",
                        help="跳过音频提取（仅提取图像）")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root) if args.dataset_root else get_default_dataset_root()
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parent / "data" / "imperfectpour"
    )

    videos_dir = dataset_root / "videos"
    annotations_dir = dataset_root / "annotations"
    train_split = dataset_root / "train_split.txt"
    val_split = dataset_root / "val_split.txt"

    if not videos_dir.exists():
        print(f"✗ 错误: videos 目录不存在: {videos_dir}")
        sys.exit(1)
    if not annotations_dir.exists():
        print(f"✗ 错误: annotations 目录不存在: {annotations_dir}")
        sys.exit(1)

    frames_dir = output_dir / "frames"

    print("=" * 60)
    print("ImperfectPour 数据预处理")
    print(f"数据集路径: {dataset_root}")
    print(f"输出路径:   {output_dir}")
    print(f"音频提取:   {'关闭' if args.skip_audio else f'开启 ({AUDIO_DURATION}s @ {AUDIO_SR}Hz)'}")
    print("=" * 60)

    frames_dir.mkdir(parents=True, exist_ok=True)

    process_recordings(train_split, output_dir / "train.csv",
                       videos_dir, annotations_dir, frames_dir, "Train",
                       skip_audio=args.skip_audio)
    process_recordings(val_split, output_dir / "val.csv",
                       videos_dir, annotations_dir, frames_dir, "Val",
                       skip_audio=args.skip_audio)

    print("\n✓ 预处理完成!")
    print(f"  Train CSV: {output_dir / 'train.csv'}")
    print(f"  Val CSV:   {output_dir / 'val.csv'}")
    print(f"  Frames:    {frames_dir}")


if __name__ == "__main__":
    main()
