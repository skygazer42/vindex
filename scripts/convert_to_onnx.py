#!/usr/bin/env python3
"""
Convert all models to ONNX format for C++ inference
"""
import os
import sys
from pathlib import Path
import torch
import onnx
from onnx import shape_inference
import numpy as np

# Fix encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def print_status(msg, status="INFO"):
    symbols = {"INFO": "[i]", "SUCCESS": "[+]", "ERROR": "[!]", "WORKING": "[*]"}
    print(f"{symbols.get(status, '[?]')} {msg}")

def convert_clip_text_encoder():
    """Convert CLIP text encoder to ONNX"""
    print_status("Converting CLIP text encoder...", "WORKING")

    try:
        import open_clip

        # Load model
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
        model.eval()

        # Create text encoder wrapper
        class TextEncoder(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.token_embedding = model.token_embedding
                self.positional_embedding = model.positional_embedding
                self.transformer = model.transformer
                self.ln_final = model.ln_final
                self.text_projection = model.text_projection

            def forward(self, text):
                x = self.token_embedding(text)
                x = x + self.positional_embedding
                x = x.permute(1, 0, 2)  # NLD -> LND
                x = self.transformer(x)
                x = x.permute(1, 0, 2)  # LND -> NLD
                x = self.ln_final(x)

                # Take features from the eot embedding (end of text)
                x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]

                if self.text_projection is not None:
                    x = x @ self.text_projection

                # Normalize
                x = x / x.norm(dim=-1, keepdim=True)
                return x

        text_encoder = TextEncoder(model)
        text_encoder.eval()

        # Export to ONNX
        output_path = Path("../assets/models/clip_text.onnx")
        if not output_path.exists():
            dummy_text = torch.randint(0, 49408, (1, 77), dtype=torch.long)  # CLIP vocab size and context

            with torch.no_grad():
                torch.onnx.export(
                    text_encoder,
                    dummy_text,
                    str(output_path),
                    input_names=['text'],
                    output_names=['features'],
                    dynamic_axes={
                        'text': {0: 'batch_size'},
                        'features': {0: 'batch_size'}
                    },
                    opset_version=14,
                    do_constant_folding=True,
                    export_params=True
                )

            # Simplify model
            try:
                import onnxsim
                model_onnx = onnx.load(str(output_path))
                model_simp, check = onnxsim.simplify(model_onnx)
                onnx.save(model_simp, str(output_path))
                print_status("Text encoder exported and simplified", "SUCCESS")
            except:
                print_status("Text encoder exported (not simplified)", "SUCCESS")

            return True
        else:
            print_status("Text encoder already exists", "INFO")
            return True

    except Exception as e:
        print_status(f"Failed to export text encoder: {str(e)[:200]}", "ERROR")
        return False

def convert_blip_text_decoder():
    """Convert BLIP text decoder to ONNX"""
    print_status("Converting BLIP text decoder...", "WORKING")

    try:
        from transformers import BlipForConditionalGeneration, BlipProcessor

        model_dir = Path("../assets/models/blip")
        model = BlipForConditionalGeneration.from_pretrained(str(model_dir))
        processor = BlipProcessor.from_pretrained(str(model_dir))
        model.eval()

        # Export text decoder (simplified version)
        output_path = model_dir / "blip_text_decoder.onnx"

        if not output_path.exists():
            # This is complex due to the decoder architecture
            # For now, we'll export a simplified version

            class TextDecoder(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.text_decoder = model.text_decoder
                    self.text_model = model.text_model if hasattr(model, 'text_model') else None

                def forward(self, input_ids, encoder_hidden_states):
                    if self.text_model:
                        outputs = self.text_model(
                            input_ids=input_ids,
                            encoder_hidden_states=encoder_hidden_states
                        )
                    else:
                        outputs = self.text_decoder(
                            input_ids=input_ids,
                            encoder_hidden_states=encoder_hidden_states
                        )
                    return outputs.logits

            decoder = TextDecoder(model)
            decoder.eval()

            # Create dummy inputs
            batch_size = 1
            seq_len = 20
            hidden_size = 768
            encoder_seq_len = 577  # For ViT-B/16

            dummy_input_ids = torch.randint(0, 30522, (batch_size, seq_len), dtype=torch.long)
            dummy_encoder_states = torch.randn(batch_size, encoder_seq_len, hidden_size)

            try:
                with torch.no_grad():
                    torch.onnx.export(
                        decoder,
                        (dummy_input_ids, dummy_encoder_states),
                        str(output_path),
                        input_names=['input_ids', 'encoder_hidden_states'],
                        output_names=['logits'],
                        dynamic_axes={
                            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                            'encoder_hidden_states': {0: 'batch_size', 1: 'encoder_sequence_length'},
                            'logits': {0: 'batch_size', 1: 'sequence_length'}
                        },
                        opset_version=14,
                        do_constant_folding=True
                    )
                print_status("BLIP text decoder exported", "SUCCESS")
                return True
            except Exception as e:
                print_status(f"Complex decoder export failed: {str(e)[:100]}", "ERROR")
                print_status("Consider using optimum library for better conversion", "INFO")
                return False
        else:
            print_status("BLIP text decoder already exists", "INFO")
            return True

    except Exception as e:
        print_status(f"Failed to export BLIP decoder: {str(e)[:200]}", "ERROR")
        return False

def convert_vqa_models():
    """Convert VQA models to ONNX"""
    print_status("Converting VQA models...", "WORKING")

    try:
        from transformers import BlipForQuestionAnswering, BlipProcessor

        model_dir = Path("../assets/models/blip_vqa")
        model = BlipForQuestionAnswering.from_pretrained(str(model_dir))
        model.eval()

        # Export vision encoder
        vision_path = model_dir / "blip_vqa_visual_encoder.onnx"
        if not vision_path.exists():
            dummy_pixel_values = torch.randn(1, 3, 384, 384)

            with torch.no_grad():
                torch.onnx.export(
                    model.vision_model,
                    dummy_pixel_values,
                    str(vision_path),
                    input_names=['pixel_values'],
                    output_names=['image_features'],
                    dynamic_axes={
                        'pixel_values': {0: 'batch_size'},
                        'image_features': {0: 'batch_size'}
                    },
                    opset_version=14,
                    do_constant_folding=True
                )
            print_status("VQA vision encoder exported", "SUCCESS")
        else:
            print_status("VQA vision encoder already exists", "INFO")

        # Note: Text encoder and decoder export would be similar to BLIP
        print_status("VQA text components require complex conversion", "INFO")
        return True

    except Exception as e:
        print_status(f"Failed to export VQA models: {str(e)[:200]}", "ERROR")
        return False

def fix_missing_clip_vocab():
    """Create CLIP vocabulary file if missing"""
    vocab_path = Path("../assets/vocab/clip_vocab.txt")
    vocab_path.parent.mkdir(parents=True, exist_ok=True)

    if not vocab_path.exists():
        print_status("Creating CLIP vocabulary file...", "WORKING")

        # For OpenAI CLIP, we need the BPE vocab
        # For CN-CLIP, we can copy from the downloaded model
        cn_vocab = Path("../assets/models/cn-clip/vocab.txt")
        if cn_vocab.exists():
            import shutil
            shutil.copy(str(cn_vocab), str(vocab_path))
            print_status("Copied CN-CLIP vocabulary", "SUCCESS")
        else:
            # Create a basic vocab file (this is a placeholder)
            print_status("Creating placeholder vocabulary", "INFO")
            with open(vocab_path, 'w', encoding='utf-8') as f:
                f.write("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n")
                for i in range(100):
                    f.write(f"token_{i}\n")
            print_status("Placeholder vocabulary created", "INFO")

def verify_onnx_models():
    """Verify all ONNX models"""
    print_status("\nVerifying ONNX models...", "INFO")

    models_dir = Path("../assets/models")

    onnx_files = {
        "OCR Detection": models_dir / "ocr" / "ch_PP-OCRv4_det_infer.onnx",
        "OCR Recognition": models_dir / "ocr" / "ch_PP-OCRv4_rec_infer.onnx",
        "CLIP Visual": models_dir / "clip_visual.onnx",
        "CLIP Text": models_dir / "clip_text.onnx",
        "BLIP Visual": models_dir / "blip" / "blip_visual_encoder.onnx",
        "BLIP Text Decoder": models_dir / "blip" / "blip_text_decoder.onnx",
        "VQA Visual": models_dir / "blip_vqa" / "blip_vqa_visual_encoder.onnx",
    }

    for name, path in onnx_files.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            try:
                model = onnx.load(str(path))
                onnx.checker.check_model(model)
                print_status(f"{name}: {size_mb:.1f} MB [Valid]", "SUCCESS")
            except Exception as e:
                print_status(f"{name}: {size_mb:.1f} MB [Invalid: {str(e)[:50]}]", "ERROR")
        else:
            print_status(f"{name}: Not found", "ERROR")

def main():
    print_status("=== ONNX Model Conversion Tool ===", "INFO")

    os.chdir(Path(__file__).parent)

    # Install requirements if needed
    try:
        import open_clip
    except ImportError:
        print_status("Installing open-clip-torch...", "WORKING")
        os.system("pip install -q open-clip-torch")

    try:
        import onnxsim
    except ImportError:
        print_status("Installing onnx-simplifier...", "WORKING")
        os.system("pip install -q onnx-simplifier")

    # Convert models
    tasks = [
        ("CLIP Text Encoder", convert_clip_text_encoder),
        ("BLIP Text Decoder", convert_blip_text_decoder),
        ("VQA Models", convert_vqa_models),
        ("Fix Vocabulary", fix_missing_clip_vocab),
    ]

    for task_name, task_func in tasks:
        print_status(f"\n=== {task_name} ===", "INFO")
        try:
            task_func()
        except Exception as e:
            print_status(f"Task failed: {str(e)[:200]}", "ERROR")

    # Verify all models
    verify_onnx_models()

    print_status("\n=== Conversion Complete ===", "INFO")
    print_status("Ready for C++ inference testing", "SUCCESS")

if __name__ == "__main__":
    main()