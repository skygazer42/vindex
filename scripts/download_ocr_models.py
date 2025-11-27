#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PP-OCRv4 模型下载工具

下载 PaddleOCR v4 的中文检测和识别模型 (ONNX 格式)
支持 VIndex C++ 应用进行 OCR 文字识别

模型来源: https://huggingface.co/SWHL/RapidOCR/tree/main/PP-OCRv4

使用方法:
    python download_ocr_models.py --output ../assets/models/ocr

模型说明:
    - ch_PP-OCRv4_det_infer.onnx: 文字检测模型，定位图像中的文字区域
    - ch_PP-OCRv4_rec_infer.onnx: 文字识别模型，识别检测到的文字内容
    - ppocr_keys_v1.txt: 识别模型的字符字典
"""

import os
import sys
import argparse
import urllib.request
import json
from pathlib import Path


# HuggingFace 模型文件 URL
HF_BASE_URL = "https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv4"

MODEL_FILES = {
    "ch_PP-OCRv4_det_infer.onnx": f"{HF_BASE_URL}/ch_PP-OCRv4_det_infer.onnx",
    "ch_PP-OCRv4_rec_infer.onnx": f"{HF_BASE_URL}/ch_PP-OCRv4_rec_infer.onnx",
}

# 字典文件 URL (从 RapidOCR 仓库)
DICT_URL = "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/ppocr_keys_v1.txt"


def download_file(url: str, output_path: str, desc: str = None):
    """下载文件并显示进度"""
    if desc:
        print(f"正在下载: {desc}")
    print(f"  URL: {url}")
    print(f"  保存到: {output_path}")

    try:
        # 创建目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 下载文件
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 // total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r  进度: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)

        urllib.request.urlretrieve(url, output_path, progress_hook)
        print()  # 换行

        # 验证文件大小
        file_size = os.path.getsize(output_path)
        print(f"  完成: {file_size / (1024 * 1024):.1f} MB")
        return True

    except Exception as e:
        print(f"  下载失败: {e}")
        return False


def create_ocr_config(output_dir: Path):
    """创建 OCR 配置文件"""
    config = {
        "model_type": "pp-ocrv4",
        "det_model": "ch_PP-OCRv4_det_infer.onnx",
        "rec_model": "ch_PP-OCRv4_rec_infer.onnx",
        "dict_file": "ppocr_keys_v1.txt",
        "det_db_thresh": 0.3,
        "det_db_box_thresh": 0.5,
        "det_db_unclip_ratio": 1.6,
        "rec_img_height": 48,
        "rec_img_width": 320,
        "max_side_len": 960,
        "use_angle_cls": False
    }

    config_path = output_dir / "ocr_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"配置文件已创建: {config_path}")


def verify_models(output_dir: Path):
    """验证下载的模型"""
    print("\n验证模型文件...")

    all_ok = True
    for filename in list(MODEL_FILES.keys()) + ["ppocr_keys_v1.txt"]:
        filepath = output_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"  ✓ {filename}: {size / (1024 * 1024):.1f} MB")
        else:
            print(f"  ✗ {filename}: 文件不存在")
            all_ok = False

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="下载 PP-OCRv4 中文 OCR 模型"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../assets/models/ocr",
        help="输出目录"
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="跳过验证"
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PP-OCRv4 中文 OCR 模型下载工具")
    print("=" * 60)
    print(f"输出目录: {output_dir.absolute()}")
    print("=" * 60)

    # 下载模型文件
    success = True
    for filename, url in MODEL_FILES.items():
        output_path = output_dir / filename
        if output_path.exists():
            print(f"\n跳过 (已存在): {filename}")
            continue

        print()
        if not download_file(url, str(output_path), filename):
            success = False

    # 下载字典文件
    dict_path = output_dir / "ppocr_keys_v1.txt"
    if not dict_path.exists():
        print()
        if not download_file(DICT_URL, str(dict_path), "字符字典"):
            success = False
    else:
        print(f"\n跳过 (已存在): ppocr_keys_v1.txt")

    # 创建配置文件
    print()
    create_ocr_config(output_dir)

    # 验证
    if not args.skip_verify:
        if not verify_models(output_dir):
            success = False

    print("\n" + "=" * 60)
    if success:
        print("下载完成!")
    else:
        print("下载完成，但有部分文件失败")
    print("=" * 60)

    print(f"\n生成的文件:")
    for f in output_dir.iterdir():
        if f.is_file():
            size = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}: {size:.1f} MB")

    print(f"\n使用方法:")
    print(f"  将 {output_dir} 目录保持在 VIndex 的 assets/models/ 下")
    print(f"  在 VIndex 中使用 OCR 功能进行文字识别")


if __name__ == "__main__":
    main()
