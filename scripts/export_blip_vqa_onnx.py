#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Taiyi-BLIP VQA 中文视觉问答模型导出工具

将 IDEA-CCNL/Taiyi-BLIP-750M-Chinese 模型的 VQA 功能导出为 ONNX 格式
支持 VIndex C++ 应用进行视觉问答

模型来源: https://huggingface.co/IDEA-CCNL/Taiyi-BLIP-750M-Chinese

使用方法:
    python export_blip_vqa_onnx.py --output ../assets/models/blip_vqa

依赖安装:
    pip install torch transformers pillow onnx onnxruntime
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from pathlib import Path


def check_dependencies():
    """检查必要依赖"""
    missing = []
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    try:
        import onnx
    except ImportError:
        missing.append("onnx")
    try:
        import onnxruntime
    except ImportError:
        missing.append("onnxruntime")
    try:
        from PIL import Image
    except ImportError:
        missing.append("pillow")

    if missing:
        print(f"缺少依赖: {', '.join(missing)}")
        print(f"请运行: pip install {' '.join(missing)}")
        sys.exit(1)


def download_model(model_name: str, cache_dir: str = None):
    """下载模型"""
    from transformers import BlipProcessor, BlipForQuestionAnswering

    print(f"正在下载 VQA 模型: {model_name}")
    print("这可能需要几分钟，请耐心等待...")

    processor = BlipProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    model = BlipForQuestionAnswering.from_pretrained(model_name, cache_dir=cache_dir)

    print("模型下载完成!")
    return processor, model


class BLIPVQAVisualEncoder(nn.Module):
    """BLIP VQA 视觉编码器"""
    def __init__(self, blip_model):
        super().__init__()
        self.vision_model = blip_model.vision_model

    def forward(self, pixel_values):
        vision_outputs = self.vision_model(pixel_values)
        image_embeds = vision_outputs.last_hidden_state
        return image_embeds


class BLIPVQATextEncoder(nn.Module):
    """BLIP VQA 文本编码器 - 编码问题"""
    def __init__(self, blip_model):
        super().__init__()
        self.text_encoder = blip_model.text_encoder

    def forward(self, input_ids, attention_mask, encoder_hidden_states):
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=True
        )
        return outputs.last_hidden_state


class BLIPVQADecoder(nn.Module):
    """BLIP VQA 文本解码器 - 生成答案"""
    def __init__(self, blip_model):
        super().__init__()
        self.text_decoder = blip_model.text_decoder

    def forward(self, input_ids, encoder_hidden_states):
        outputs = self.text_decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=True
        )
        return outputs.logits


def export_visual_encoder(model, output_dir: Path, opset_version: int = 14):
    """导出视觉编码器"""
    print("\n正在导出 VQA 视觉编码器...")

    visual_encoder = BLIPVQAVisualEncoder(model)
    visual_encoder.eval()

    dummy_input = torch.randn(1, 3, 384, 384)
    output_path = output_dir / "blip_vqa_visual_encoder.onnx"

    with torch.no_grad():
        torch.onnx.export(
            visual_encoder,
            dummy_input,
            str(output_path),
            opset_version=opset_version,
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "image_embeds": {0: "batch_size", 1: "sequence_length"}
            }
        )

    print(f"VQA 视觉编码器已导出: {output_path}")
    return output_path


def export_text_encoder(model, output_dir: Path, opset_version: int = 14):
    """导出文本编码器（编码问题）"""
    print("\n正在导出 VQA 文本编码器...")

    text_encoder = BLIPVQATextEncoder(model)
    text_encoder.eval()

    batch_size = 1
    seq_len = 32
    encoder_seq_len = 577
    hidden_size = model.config.text_config.hidden_size

    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    dummy_encoder_hidden = torch.randn(batch_size, encoder_seq_len, hidden_size)

    output_path = output_dir / "blip_vqa_text_encoder.onnx"

    with torch.no_grad():
        torch.onnx.export(
            text_encoder,
            (dummy_input_ids, dummy_attention_mask, dummy_encoder_hidden),
            str(output_path),
            opset_version=opset_version,
            input_names=["input_ids", "attention_mask", "encoder_hidden_states"],
            output_names=["question_embeds"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "encoder_hidden_states": {0: "batch_size", 1: "encoder_sequence_length"},
                "question_embeds": {0: "batch_size", 1: "sequence_length"}
            }
        )

    print(f"VQA 文本编码器已导出: {output_path}")
    return output_path


def export_text_decoder(model, output_dir: Path, opset_version: int = 14):
    """导出文本解码器（生成答案）"""
    print("\n正在导出 VQA 文本解码器...")

    text_decoder = BLIPVQADecoder(model)
    text_decoder.eval()

    batch_size = 1
    seq_len = 16
    encoder_seq_len = 32
    hidden_size = model.config.text_config.hidden_size

    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    dummy_encoder_hidden = torch.randn(batch_size, encoder_seq_len, hidden_size)

    output_path = output_dir / "blip_vqa_text_decoder.onnx"

    with torch.no_grad():
        torch.onnx.export(
            text_decoder,
            (dummy_input_ids, dummy_encoder_hidden),
            str(output_path),
            opset_version=opset_version,
            input_names=["input_ids", "encoder_hidden_states"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "encoder_hidden_states": {0: "batch_size", 1: "encoder_sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"}
            }
        )

    print(f"VQA 文本解码器已导出: {output_path}")
    return output_path


def save_vocab_and_config(processor, model, output_dir: Path):
    """保存词表和配置文件"""
    print("\n正在保存词表和配置...")

    # 保存 tokenizer
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(exist_ok=True)
    processor.tokenizer.save_pretrained(str(tokenizer_dir))

    # 保存配置
    config = {
        "model_type": "blip_vqa",
        "model_name": "Taiyi-BLIP-750M-Chinese-VQA",
        "image_size": 384,
        "patch_size": 16,
        "hidden_size": model.config.text_config.hidden_size,
        "vocab_size": model.config.text_config.vocab_size,
        "max_question_length": 32,
        "max_answer_length": 16,
        "bos_token_id": processor.tokenizer.bos_token_id or processor.tokenizer.cls_token_id,
        "eos_token_id": processor.tokenizer.eos_token_id or processor.tokenizer.sep_token_id,
        "pad_token_id": processor.tokenizer.pad_token_id,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711]
    }

    config_path = output_dir / "blip_vqa_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"配置已保存: {config_path}")
    print(f"词表已保存: {tokenizer_dir}")


def verify_onnx_model(onnx_path: Path):
    """验证 ONNX 模型"""
    import onnx
    import onnxruntime as ort

    print(f"\n正在验证: {onnx_path.name}")

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    print(f"  ✓ 模型结构有效")

    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])

    inputs = {}
    for inp in session.get_inputs():
        shape = [s if isinstance(s, int) else 1 for s in inp.shape]
        if inp.type == 'tensor(float)':
            inputs[inp.name] = torch.randn(*shape).numpy()
        elif inp.type == 'tensor(int64)':
            inputs[inp.name] = torch.randint(0, 1000, shape).numpy()

    outputs = session.run(None, inputs)
    print(f"  ✓ 推理测试通过")
    print(f"  输出形状: {[o.shape for o in outputs]}")


def test_vqa(model, processor, image_path: str = None):
    """测试 VQA 功能"""
    from PIL import Image
    import requests
    from io import BytesIO

    print("\n测试视觉问答...")

    if image_path and os.path.exists(image_path):
        image = Image.open(image_path).convert("RGB")
    else:
        url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
        try:
            response = requests.get(url, timeout=10)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except:
            print("  跳过测试（无法获取测试图像）")
            return

    question = "这是什么?"

    inputs = processor(images=image, text=question, return_tensors="pt")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_length=20)

    answer = processor.decode(generated_ids[0], skip_special_tokens=True)
    print(f"  问题: {question}")
    print(f"  答案: {answer}")


def main():
    parser = argparse.ArgumentParser(
        description="导出 Taiyi-BLIP VQA 中文视觉问答模型为 ONNX 格式"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="IDEA-CCNL/Taiyi-BLIP-750M-Chinese",
        help="Hugging Face 模型名称"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../assets/models/blip_vqa",
        help="输出目录"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="模型缓存目录"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset 版本"
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="跳过 ONNX 模型验证"
    )
    parser.add_argument(
        "--test-image",
        type=str,
        default=None,
        help="测试图像路径"
    )

    args = parser.parse_args()

    check_dependencies()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Taiyi-BLIP VQA 中文视觉问答模型导出工具")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"输出目录: {output_dir.absolute()}")
    print("=" * 60)

    processor, model = download_model(args.model, args.cache_dir)
    model.eval()

    test_vqa(model, processor, args.test_image)

    visual_path = export_visual_encoder(model, output_dir, args.opset)
    encoder_path = export_text_encoder(model, output_dir, args.opset)
    decoder_path = export_text_decoder(model, output_dir, args.opset)

    save_vocab_and_config(processor, model, output_dir)

    if not args.skip_verify:
        print("\n" + "=" * 60)
        print("验证导出的模型")
        print("=" * 60)
        verify_onnx_model(visual_path)
        verify_onnx_model(encoder_path)
        verify_onnx_model(decoder_path)

    print("\n" + "=" * 60)
    print("导出完成!")
    print("=" * 60)
    print(f"\n生成的文件:")
    for f in output_dir.iterdir():
        if f.is_file():
            size = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}: {size:.1f} MB")
        elif f.is_dir():
            print(f"  {f.name}/ (目录)")

    print(f"\n使用方法:")
    print(f"  将 {output_dir} 目录复制到 VIndex 的 assets/models/ 下")
    print(f"  在 VIndex 中使用 VQA 功能进行视觉问答")


if __name__ == "__main__":
    main()
