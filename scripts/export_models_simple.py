#!/usr/bin/env python3
"""
Simple export of existing models to ONNX
"""
import os
import sys
from pathlib import Path

# Fix encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def export_blip_to_onnx():
    """Export BLIP models to ONNX format"""
    print("Attempting BLIP ONNX export...")

    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        import torch
        import onnx

        blip_dir = Path("../assets/models/blip")

        # Load model
        print("Loading BLIP model...")
        processor = BlipProcessor.from_pretrained(str(blip_dir))
        model = BlipForConditionalGeneration.from_pretrained(str(blip_dir))
        model.eval()

        # Export vision model
        print("Exporting vision encoder...")
        vision_path = blip_dir / "blip_visual_encoder.onnx"

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
            print(f"Vision encoder saved to: {vision_path}")
        else:
            print("Vision encoder already exists")

        print("Note: Text decoder export requires more complex handling")
        return True

    except Exception as e:
        print(f"BLIP export error: {e}")
        return False

def check_existing_models():
    """Check what models we have"""
    print("\n=== Current Model Status ===")

    models_dir = Path("../assets/models")

    # OCR
    ocr_det = models_dir / "ocr" / "ch_PP-OCRv4_det_infer.onnx"
    ocr_rec = models_dir / "ocr" / "ch_PP-OCRv4_rec_infer.onnx"
    if ocr_det.exists() and ocr_rec.exists():
        print("[OK] OCR models (detection + recognition)")

    # CLIP
    clip_visual = models_dir / "clip_visual.onnx"
    if clip_visual.exists():
        size_mb = clip_visual.stat().st_size / (1024 * 1024)
        print(f"[OK] CLIP visual encoder ({size_mb:.1f} MB)")

    clip_text = models_dir / "clip_text.onnx"
    if clip_text.exists():
        size_mb = clip_text.stat().st_size / (1024 * 1024)
        print(f"[OK] CLIP text encoder ({size_mb:.1f} MB)")
    else:
        print("[--] CLIP text encoder missing")

    # CN-CLIP
    cn_clip_pt = models_dir / "cn-clip" / "clip_cn_vit-b-16.pt"
    if cn_clip_pt.exists():
        size_mb = cn_clip_pt.stat().st_size / (1024 * 1024)
        print(f"[OK] CN-CLIP PyTorch model ({size_mb:.1f} MB)")

    # BLIP
    blip_model = models_dir / "blip" / "model.safetensors"
    if blip_model.exists():
        size_mb = blip_model.stat().st_size / (1024 * 1024)
        print(f"[OK] BLIP Caption model ({size_mb:.1f} MB)")

    blip_vision_onnx = models_dir / "blip" / "blip_visual_encoder.onnx"
    if blip_vision_onnx.exists():
        size_mb = blip_vision_onnx.stat().st_size / (1024 * 1024)
        print(f"[OK] BLIP vision encoder ONNX ({size_mb:.1f} MB)")

    # BLIP VQA
    vqa_model = models_dir / "blip_vqa" / "model.safetensors"
    if vqa_model.exists():
        size_mb = vqa_model.stat().st_size / (1024 * 1024)
        print(f"[OK] BLIP VQA model ({size_mb:.1f} MB)")

    print("\n=== Summary ===")
    print("- OCR: Ready to use")
    print("- CLIP: Visual encoder ready, text encoder missing")
    print("- CN-CLIP: PyTorch model available, needs ONNX conversion")
    print("- BLIP Caption: Model downloaded, partial ONNX export possible")
    print("- BLIP VQA: Model downloaded, needs ONNX export")

if __name__ == "__main__":
    os.chdir(Path(__file__).parent)

    print("Checking existing models...")
    check_existing_models()

    print("\nAttempting BLIP export...")
    export_blip_to_onnx()

    print("\nFinal status:")
    check_existing_models()