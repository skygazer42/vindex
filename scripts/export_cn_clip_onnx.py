#!/usr/bin/env python3
"""
Export CN-CLIP to ONNX format
"""
import os
import sys
import torch
import onnx
from pathlib import Path

# Fix encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def export_cn_clip():
    print("Exporting CN-CLIP to ONNX...")

    model_path = Path("../assets/models/cn-clip/clip_cn_vit-b-16.pt")

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return False

    try:
        # Load CN-CLIP model
        print("Loading CN-CLIP model...")
        import cn_clip.clip as clip
        from cn_clip.clip import load_from_name

        device = "cpu"
        model, preprocess = load_from_name("ViT-B-16", device=device, download_root='../assets/models/cn-clip/')
        model.eval()

        # Export visual encoder
        print("Exporting visual encoder...")
        visual_path = Path("../assets/models/clip_visual_cn.onnx")

        if not visual_path.exists():
            dummy_image = torch.randn(1, 3, 224, 224)

            # Extract just the visual part
            with torch.no_grad():
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
            print(f"Visual encoder saved to: {visual_path}")
        else:
            print("Visual encoder already exists")

        # Export text encoder
        print("Exporting text encoder...")
        text_path = Path("../assets/models/clip_text_cn.onnx")

        if not text_path.exists():
            # CN-CLIP uses different tokenizer
            dummy_text = torch.randint(0, 21128, (1, 52))  # CN-CLIP uses 52 context length

            with torch.no_grad():
                # Create a wrapper for text encoding
                class TextEncoder(torch.nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model

                    def forward(self, text):
                        x = self.model.token_embedding(text)
                        x = x + self.model.positional_embedding
                        x = x.permute(1, 0, 2)  # NLD -> LND
                        x = self.model.transformer(x)
                        x = x.permute(1, 0, 2)  # LND -> NLD
                        x = self.model.ln_final(x)
                        # Take features from the eot embedding
                        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
                        x = x @ self.model.text_projection
                        return x

                text_encoder = TextEncoder(model)
                text_encoder.eval()

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
            print(f"Text encoder saved to: {text_path}")
        else:
            print("Text encoder already exists")

        print("CN-CLIP export completed!")
        return True

    except ImportError:
        print("CN-CLIP not installed. Installing...")
        os.system("pip install cn-clip")
        return export_cn_clip()  # Retry after install

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    success = export_cn_clip()
    sys.exit(0 if success else 1)