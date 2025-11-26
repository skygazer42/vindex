#!/usr/bin/env python3
"""
下载并准备中文CLIP模型

支持的模型:
1. CN-CLIP (OFA-Sys) - 推荐，中英双语
2. Taiyi-CLIP (IDEA-CCNL) - 纯中文优化

使用方法:
    python download_chinese_clip.py --model cn-clip
    python download_chinese_clip.py --model taiyi
    python download_chinese_clip.py --model both
"""

import os
import sys
import argparse
from pathlib import Path
import json

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False
    print("⚠️  huggingface_hub未安装，部分功能不可用")
    print("   安装: pip install huggingface_hub")


def print_header(text):
    """打印标题"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def download_cn_clip(output_dir, export_onnx=False, use_mirror=False):
    """下载CN-CLIP模型"""
    print_header("📥 下载 CN-CLIP 模型")

    if not HAS_HF_HUB:
        print("❌ 需要安装 huggingface_hub")
        print("   pip install huggingface_hub")
        return None

    output_path = Path(output_dir) / "cn-clip"
    output_path.mkdir(parents=True, exist_ok=True)

    # 选择仓库
    if use_mirror:
        repo_id = "eisneim/cn-clip_vit-b-16"
        print("📦 模型信息:")
        print("   名称: CN-CLIP ViT-B/16 (镜像版本)")
        print("   来源: eisneim/cn-clip_vit-b-16")
    else:
        repo_id = "OFA-Sys/chinese-clip-vit-base-patch16"
        print("📦 模型信息:")
        print("   名称: CN-CLIP ViT-B/16")
        print("   来源: OFA-Sys/chinese-clip-vit-base-patch16")

    print("   大小: ~600MB")
    print("   特征维度: 512")
    print("   语言: 中英双语")
    print()

    try:
        print("⏬ 开始下载...")
        print("   (首次下载可能需要几分钟，请耐心等待)")

        # 下载完整仓库
        print(f"\n📂 下载到: {output_path}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=output_path,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.msgpack", "*.h5", ".gitattributes"]
        )

        print("\n✅ CN-CLIP 模型下载完成!")
        print(f"   保存位置: {output_path}")

        # 保存模型信息
        model_info = {
            "name": "CN-CLIP",
            "repo": repo_id,
            "type": "chinese-clip",
            "embedding_dim": 512,
            "language": ["zh", "en"],
            "visual_encoder": "ViT-B/16",
            "text_encoder": "BERT-base-chinese"
        }

        with open(output_path / "model_info.json", "w", encoding="utf-8") as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)

        # 可选: 导出为ONNX
        if export_onnx:
            print("\n🔄 准备导出为ONNX格式...")
            print("   请运行: python export_cn_clip_to_onnx.py")
            print("   (需要安装 PyTorch 和 cn_clip)")

        return output_path

    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None


def download_taiyi_clip(output_dir, export_onnx=False):
    """下载Taiyi-CLIP模型"""
    print_header("📥 下载 Taiyi-CLIP 模型")

    if not HAS_HF_HUB:
        print("❌ 需要安装 huggingface_hub")
        return None

    output_path = Path(output_dir) / "taiyi-clip"
    output_path.mkdir(parents=True, exist_ok=True)

    print("📦 模型信息:")
    print("   名称: Taiyi-CLIP-Roberta-102M")
    print("   来源: IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese")
    print("   大小: ~400MB")
    print("   特征维度: 512")
    print("   语言: 中文")
    print()

    try:
        print("⏬ 开始下载...")

        repo_id = "IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese"

        # 下载模型
        model_path = snapshot_download(
            repo_id=repo_id,
            local_dir=output_path,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.msgpack", "*.h5"]  # 忽略不需要的格式
        )

        print(f"\n✅ Taiyi-CLIP 模型下载完成!")
        print(f"   保存位置: {output_path}")

        # 保存模型信息
        model_info = {
            "name": "Taiyi-CLIP",
            "repo": repo_id,
            "type": "taiyi-clip",
            "embedding_dim": 512,
            "language": ["zh"],
            "visual_encoder": "ViT-B/16",
            "text_encoder": "Chinese-Roberta-wwm-ext"
        }

        with open(output_path / "model_info.json", "w", encoding="utf-8") as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)

        if export_onnx:
            print("\n🔄 准备导出为ONNX格式...")
            print("   需要手动转换（模型结构特殊）")

        return output_path

    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None


def check_requirements():
    """检查依赖"""
    print_header("🔍 检查依赖")

    requirements = {
        "huggingface_hub": HAS_HF_HUB,
    }

    all_ok = True
    for pkg, installed in requirements.items():
        status = "✅" if installed else "❌"
        print(f"   {status} {pkg}")
        if not installed:
            all_ok = False

    if not all_ok:
        print("\n💡 安装缺失依赖:")
        print("   pip install huggingface_hub")
        print()

    return all_ok


def print_summary(downloaded_models):
    """打印总结"""
    print_header("📊 下载总结")

    if not downloaded_models:
        print("   ❌ 没有成功下载任何模型")
        return

    print(f"   ✅ 成功下载 {len(downloaded_models)} 个模型:\n")

    for model_name, model_path in downloaded_models.items():
        print(f"   📁 {model_name}")
        print(f"      位置: {model_path}")

        # 读取模型信息
        info_file = Path(model_path) / "model_info.json"
        if info_file.exists():
            with open(info_file, encoding="utf-8") as f:
                info = json.load(f)
                print(f"      语言: {', '.join(info['language'])}")
                print(f"      维度: {info['embedding_dim']}")
        print()

    print("📝 下一步:")
    print("   1. 转换为ONNX格式（用于VIndex）:")
    print("      python export_cn_clip_to_onnx.py")
    print()
    print("   2. 或者查看详细文档:")
    print("      docs/CHINESE_CLIP_SUPPORT.md")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="下载中文CLIP模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载CN-CLIP（推荐）
  python download_chinese_clip.py --model cn-clip

  # 下载Taiyi-CLIP
  python download_chinese_clip.py --model taiyi

  # 下载两个模型
  python download_chinese_clip.py --model both

  # 指定输出目录
  python download_chinese_clip.py --model cn-clip --output ../models
        """
    )

    parser.add_argument(
        "--model",
        choices=["cn-clip", "taiyi", "both"],
        default="cn-clip",
        help="选择要下载的模型 (默认: cn-clip)"
    )

    parser.add_argument(
        "--output",
        default="../assets/models",
        help="输出目录 (默认: ../assets/models)"
    )

    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="下载后尝试导出为ONNX格式（需要额外依赖）"
    )

    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="跳过依赖检查"
    )

    parser.add_argument(
        "--mirror",
        action="store_true",
        help="使用镜像仓库（eisneim版本）"
    )

    args = parser.parse_args()

    print("🌏 中文CLIP模型下载工具")
    print()

    # 检查依赖
    if not args.skip_check:
        if not check_requirements():
            sys.exit(1)

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📂 输出目录: {output_dir.absolute()}\n")

    downloaded_models = {}

    # 下载模型
    if args.model in ["cn-clip", "both"]:
        model_path = download_cn_clip(output_dir, args.export_onnx, args.mirror)
        if model_path:
            repo_name = "CN-CLIP (镜像)" if args.mirror else "CN-CLIP"
            downloaded_models[repo_name] = model_path

    if args.model in ["taiyi", "both"]:
        model_path = download_taiyi_clip(output_dir, args.export_onnx)
        if model_path:
            downloaded_models["Taiyi-CLIP"] = model_path

    # 打印总结
    print_summary(downloaded_models)

    if downloaded_models:
        print("🎉 全部完成！")
        sys.exit(0)
    else:
        print("❌ 下载失败，请检查网络连接和依赖")
        sys.exit(1)


if __name__ == "__main__":
    main()
