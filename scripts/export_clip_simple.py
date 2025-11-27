#!/usr/bin/env python3
"""
Export CLIP models to ONNX format (simplified version for Windows)
Uses OpenCLIP and transformers instead of cn-clip
"""

import os
import sys
from pathlib import Path

# Fix encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def export_openclip():
    """Export OpenAI CLIP via open_clip to ONNX"""
    print("=== Exporting OpenCLIP ViT-B/16 ===")

    try:
        import torch
        import open_clip

        # Create output directory
        output_dir = Path("../assets/models")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load model
        print("Loading OpenCLIP ViT-B/16...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-16',
            pretrained='openai'
        )
        model.eval()

        # Export visual encoder
        visual_path = output_dir / "clip_visual.onnx"
        if not visual_path.exists():
            print(f"Exporting visual encoder to {visual_path}...")
            dummy_image = torch.randn(1, 3, 224, 224)

            torch.onnx.export(
                model.visual,
                dummy_image,
                str(visual_path),
                input_names=['image'],
                output_names=['features'],
                dynamic_axes={
                    'image': {0: 'batch_size'},
                    'features': {0: 'batch_size'}
                },
                opset_version=14,
                do_constant_folding=True
            )
            print(f"Visual encoder saved: {visual_path}")
        else:
            print(f"Visual encoder already exists: {visual_path}")

        # Export text encoder
        text_path = output_dir / "clip_text.onnx"
        if not text_path.exists():
            print(f"Exporting text encoder to {text_path}...")

            # Create wrapper for text encoder
            class TextEncoderWrapper(torch.nn.Module):
                def __init__(self, clip_model):
                    super().__init__()
                    self.token_embedding = clip_model.token_embedding
                    self.positional_embedding = clip_model.positional_embedding
                    self.transformer = clip_model.transformer
                    self.ln_final = clip_model.ln_final
                    self.text_projection = clip_model.text_projection
                    self.attn_mask = clip_model.attn_mask

                def forward(self, text):
                    x = self.token_embedding(text)
                    x = x + self.positional_embedding
                    x = x.permute(1, 0, 2)  # NLD -> LND
                    x = self.transformer(x, attn_mask=self.attn_mask)
                    x = x.permute(1, 0, 2)  # LND -> NLD
                    x = self.ln_final(x)
                    # Take features from EOT token
                    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
                    x = x @ self.text_projection
                    return x

            text_encoder = TextEncoderWrapper(model)
            text_encoder.eval()

            dummy_text = torch.randint(0, 49408, (1, 77))  # OpenCLIP vocab size

            torch.onnx.export(
                text_encoder,
                dummy_text,
                str(text_path),
                input_names=['text'],
                output_names=['features'],
                dynamic_axes={
                    'text': {0: 'batch_size'},
                    'features': {0: 'batch_size'}
                },
                opset_version=14,
                do_constant_folding=True
            )
            print(f"Text encoder saved: {text_path}")
        else:
            print(f"Text encoder already exists: {text_path}")

        return True

    except Exception as e:
        print(f"Error exporting OpenCLIP: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_chinese_clip_transformers():
    """Export Chinese CLIP via transformers to ONNX"""
    print("\n=== Exporting Chinese CLIP via Transformers ===")

    try:
        import torch
        from transformers import ChineseCLIPModel, ChineseCLIPProcessor

        output_dir = Path("../assets/models/cn-clip-eisneim")
        output_dir.mkdir(parents=True, exist_ok=True)

        print("Loading Chinese CLIP from HuggingFace...")
        model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
        processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
        model.eval()

        # Export visual encoder
        visual_path = output_dir / "vit-b-16.img.fp32.onnx"
        if not visual_path.exists():
            print(f"Exporting visual encoder to {visual_path}...")

            class VisualEncoder(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.vision_model = model.vision_model
                    self.visual_projection = model.visual_projection

                def forward(self, pixel_values):
                    vision_outputs = self.vision_model(pixel_values=pixel_values)
                    pooled_output = vision_outputs.pooler_output
                    image_features = self.visual_projection(pooled_output)
                    return image_features

            visual_encoder = VisualEncoder(model)
            visual_encoder.eval()

            dummy_image = torch.randn(1, 3, 224, 224)

            torch.onnx.export(
                visual_encoder,
                dummy_image,
                str(visual_path),
                input_names=['pixel_values'],
                output_names=['features'],
                dynamic_axes={
                    'pixel_values': {0: 'batch_size'},
                    'features': {0: 'batch_size'}
                },
                opset_version=14,
                do_constant_folding=True
            )
            print(f"Visual encoder saved: {visual_path}")
        else:
            print(f"Visual encoder already exists: {visual_path}")

        # Export text encoder
        text_path = output_dir / "vit-b-16.txt.fp32.onnx"
        if not text_path.exists():
            print(f"Exporting text encoder to {text_path}...")

            class TextEncoder(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.text_model = model.text_model
                    self.text_projection = model.text_projection

                def forward(self, input_ids, attention_mask):
                    text_outputs = self.text_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                    # Chinese CLIP may return None for pooler_output, use last_hidden_state[:, 0] instead
                    pooled_output = text_outputs.pooler_output
                    if pooled_output is None:
                        # Use [CLS] token output
                        pooled_output = text_outputs.last_hidden_state[:, 0]
                    text_features = self.text_projection(pooled_output)
                    return text_features

            text_encoder = TextEncoder(model)
            text_encoder.eval()

            # Chinese CLIP uses 52 token context length
            dummy_input_ids = torch.randint(0, 21128, (1, 52))
            dummy_attention_mask = torch.ones(1, 52, dtype=torch.long)

            torch.onnx.export(
                text_encoder,
                (dummy_input_ids, dummy_attention_mask),
                str(text_path),
                input_names=['input_ids', 'attention_mask'],
                output_names=['features'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size'},
                    'attention_mask': {0: 'batch_size'},
                    'features': {0: 'batch_size'}
                },
                opset_version=14,
                do_constant_folding=True
            )
            print(f"Text encoder saved: {text_path}")
        else:
            print(f"Text encoder already exists: {text_path}")

        # Save vocab for text tokenizer
        vocab_dir = Path("../assets/vocab")
        vocab_dir.mkdir(parents=True, exist_ok=True)
        vocab_path = vocab_dir / "clip_vocab_cn.txt"

        if not vocab_path.exists():
            print(f"Saving vocabulary to {vocab_path}...")
            tokenizer = processor.tokenizer
            vocab = tokenizer.get_vocab()
            sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
            with open(vocab_path, 'w', encoding='utf-8') as f:
                for token, idx in sorted_vocab:
                    f.write(f"{token}\n")
            print(f"Vocabulary saved: {vocab_path}")

        return True

    except Exception as e:
        print(f"Error exporting Chinese CLIP: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_models():
    """Verify exported models"""
    print("\n=== Verifying Exported Models ===")

    models_dir = Path("../assets/models")

    models_to_check = [
        ("OpenCLIP Visual", models_dir / "clip_visual.onnx"),
        ("OpenCLIP Text", models_dir / "clip_text.onnx"),
        ("CN-CLIP Visual", models_dir / "cn-clip-eisneim/vit-b-16.img.fp32.onnx"),
        ("CN-CLIP Text", models_dir / "cn-clip-eisneim/vit-b-16.txt.fp32.onnx"),
        ("OCR Detection", models_dir / "ocr/ch_PP-OCRv4_det_infer.onnx"),
        ("OCR Recognition", models_dir / "ocr/ch_PP-OCRv4_rec_infer.onnx"),
    ]

    for name, path in models_to_check:
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"[OK] {name}: {path.name} ({size_mb:.1f} MB)")
        else:
            print(f"[MISSING] {name}: {path}")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent)

    print("CLIP Model Export Tool for VIndex")
    print("=" * 50)

    # Export OpenCLIP (for English)
    export_openclip()

    # Export Chinese CLIP (for Chinese text search)
    export_chinese_clip_transformers()

    # Verify
    verify_models()

    print("\n" + "=" * 50)
    print("Export complete!")
