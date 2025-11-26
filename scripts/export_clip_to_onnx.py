#!/usr/bin/env python3
"""
CLIP Model Export to ONNX
将 CLIP 模型导出为 ONNX 格式用于 C++ 推理
支持 OpenAI CLIP 和 OpenCLIP
"""

import argparse
import json
from pathlib import Path
import torch
import onnx
from onnx import shape_inference
import open_clip


class CLIPExporter:
    """CLIP模型ONNX导出器"""

    def __init__(self, model_name='ViT-L-14', pretrained='openai', output_dir='../assets/models'):
        self.model_name = model_name
        self.pretrained = pretrained
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading CLIP model: {model_name} ({pretrained})")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def export_visual_encoder(self):
        """导出视觉编码器"""
        print("\n=== Exporting Visual Encoder ===")

        # 准备示例输入（CLIP默认224x224，归一化后的RGB图像）
        dummy_image = torch.randn(1, 3, 224, 224)

        # 创建仅包含视觉编码器的模块
        class VisualEncoder(torch.nn.Module):
            def __init__(self, clip_model):
                super().__init__()
                self.visual = clip_model.visual

            def forward(self, image):
                features = self.visual(image)
                # L2归一化
                features = features / features.norm(dim=-1, keepdim=True)
                return features

        visual_model = VisualEncoder(self.model)
        visual_model.eval()

        output_path = self.output_dir / "clip_visual.onnx"

        with torch.no_grad():
            torch.onnx.export(
                visual_model,
                dummy_image,
                str(output_path),
                input_names=['image'],
                output_names=['image_features'],
                dynamic_axes={
                    'image': {0: 'batch_size'},
                    'image_features': {0: 'batch_size'}
                },
                opset_version=14,
                do_constant_folding=True
            )

        # 优化模型
        self._optimize_onnx(output_path)

        # 验证输出
        self._validate_model(output_path, dummy_image, visual_model)

        print(f"✓ Visual encoder saved to: {output_path}")
        return output_path

    def export_text_encoder(self):
        """导出文本编码器"""
        print("\n=== Exporting Text Encoder ===")

        # CLIP文本输入是token IDs（通常最大长度77）
        dummy_text_tokens = torch.randint(0, 49408, (1, 77))  # CLIP词表大小49408

        class TextEncoder(torch.nn.Module):
            def __init__(self, clip_model):
                super().__init__()
                self.token_embedding = clip_model.token_embedding
                self.positional_embedding = clip_model.positional_embedding
                self.transformer = clip_model.transformer
                self.ln_final = clip_model.ln_final
                self.text_projection = clip_model.text_projection

            def forward(self, text_tokens):
                x = self.token_embedding(text_tokens)
                x = x + self.positional_embedding
                x = x.permute(1, 0, 2)  # NLD -> LND
                x = self.transformer(x)
                x = x.permute(1, 0, 2)  # LND -> NLD
                x = self.ln_final(x)

                # 提取[EOS] token的特征
                # text_tokens.argmax(dim=-1) 找到最后一个非0的位置
                x = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ self.text_projection

                # L2归一化
                x = x / x.norm(dim=-1, keepdim=True)
                return x

        text_model = TextEncoder(self.model)
        text_model.eval()

        output_path = self.output_dir / "clip_text.onnx"

        with torch.no_grad():
            torch.onnx.export(
                text_model,
                dummy_text_tokens,
                str(output_path),
                input_names=['text_tokens'],
                output_names=['text_features'],
                dynamic_axes={
                    'text_tokens': {0: 'batch_size'},
                    'text_features': {0: 'batch_size'}
                },
                opset_version=14,
                do_constant_folding=True
            )

        self._optimize_onnx(output_path)
        self._validate_model(output_path, dummy_text_tokens, text_model)

        print(f"✓ Text encoder saved to: {output_path}")
        return output_path

    def export_vocab(self):
        """导出词表和配置"""
        print("\n=== Exporting Vocabulary ===")

        # 获取BPE编码器的词表
        vocab_path = self.output_dir.parent / "vocab" / "clip_vocab.json"
        vocab_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存词表配置
        vocab_config = {
            "model_name": self.model_name,
            "vocab_size": 49408,
            "context_length": 77,
            "bpe_path": str(self.output_dir.parent / "vocab" / "bpe_simple_vocab_16e6.txt")
        }

        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_config, f, indent=2)

        # 下载BPE词表文件（如果需要）
        bpe_url = "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"
        print(f"Note: Download BPE vocab from: {bpe_url}")
        print(f"      Extract to: {vocab_config['bpe_path']}")

        print(f"✓ Vocabulary config saved to: {vocab_path}")
        return vocab_path

    def _optimize_onnx(self, model_path):
        """优化ONNX模型（类型推断、形状推断）"""
        model = onnx.load(str(model_path))
        model = shape_inference.infer_shapes(model)
        onnx.save(model, str(model_path))

    def _validate_model(self, onnx_path, dummy_input, torch_model):
        """验证ONNX模型输出与PyTorch一致"""
        import onnxruntime as ort

        # PyTorch输出
        with torch.no_grad():
            torch_output = torch_model(dummy_input).numpy()

        # ONNX Runtime输出
        session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        onnx_output = session.run(None, {input_name: dummy_input.numpy()})[0]

        # 比较差异
        diff = abs(torch_output - onnx_output).max()
        print(f"  Max difference: {diff:.6f}")

        if diff < 1e-4:
            print("  ✓ Validation passed!")
        else:
            print(f"  ⚠ Warning: Large difference detected ({diff})")

    def export_all(self):
        """导出所有组件"""
        print("="*50)
        print(f"Exporting CLIP {self.model_name} to ONNX")
        print("="*50)

        visual_path = self.export_visual_encoder()
        text_path = self.export_text_encoder()
        vocab_path = self.export_vocab()

        # 生成模型信息文件
        info = {
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "visual_encoder": str(visual_path.name),
            "text_encoder": str(text_path.name),
            "vocab_config": str(vocab_path),
            "embedding_dim": 768 if 'L' in self.model_name else 512,
            "image_size": 224,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711]
        }

        info_path = self.output_dir / "model_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)

        print("\n" + "="*50)
        print("Export Complete!")
        print("="*50)
        print(f"Visual Model: {visual_path}")
        print(f"Text Model:   {text_path}")
        print(f"Model Info:   {info_path}")
        print("\nNext steps:")
        print("1. Download BPE vocabulary file")
        print("2. Test models with ONNX Runtime")
        print("3. Integrate into C++ application")


def main():
    parser = argparse.ArgumentParser(description='Export CLIP models to ONNX')
    parser.add_argument('--model', default='ViT-L-14',
                        help='CLIP model name (e.g., ViT-B-32, ViT-L-14)')
    parser.add_argument('--pretrained', default='openai',
                        help='Pretrained weights (openai, laion400m, etc.)')
    parser.add_argument('--output', default='../assets/models',
                        help='Output directory')

    args = parser.parse_args()

    exporter = CLIPExporter(
        model_name=args.model,
        pretrained=args.pretrained,
        output_dir=args.output
    )

    exporter.export_all()


if __name__ == '__main__':
    main()
