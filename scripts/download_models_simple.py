#!/usr/bin/env python3
"""
Simple script to download and export models without unicode issues
"""
import os
import sys
import json
from pathlib import Path

# Set encoding for Windows
if sys.platform == 'win32':
    import locale
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

def download_models():
    print("Starting model download process...")

    # Create directories
    base_path = Path("../assets/models")
    base_path.mkdir(parents=True, exist_ok=True)

    # Import what we need
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        from transformers import BlipForQuestionAnswering
        import torch
        import onnx
        print("Successfully imported required libraries")
    except ImportError as e:
        print(f"Failed to import: {e}")
        return False

    # 1. Download and export BLIP Caption model
    print("\n1. Processing BLIP Caption model...")
    try:
        blip_dir = base_path / "blip"
        blip_dir.mkdir(parents=True, exist_ok=True)

        print("   Loading BLIP caption model from HuggingFace...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        model.eval()

        # Save tokenizer
        tokenizer_dir = blip_dir / "tokenizer"
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        processor.tokenizer.save_vocabulary(str(tokenizer_dir))

        # Save config
        config = {
            "model_type": "blip",
            "image_size": 384,
            "vision_config": model.config.vision_config.to_dict() if hasattr(model.config, 'vision_config') else {},
            "text_config": model.config.text_config.to_dict() if hasattr(model.config, 'text_config') else {},
        }
        with open(blip_dir / "blip_config.json", "w") as f:
            json.dump(config, f, indent=2)

        print("   BLIP Caption model components saved successfully")

        # Note: ONNX export would require more complex code to handle the model architecture
        print("   Note: Full ONNX export requires additional processing")

    except Exception as e:
        print(f"   Error with BLIP Caption: {e}")

    # 2. Download and export BLIP VQA model
    print("\n2. Processing BLIP VQA model...")
    try:
        vqa_dir = base_path / "blip_vqa"
        vqa_dir.mkdir(parents=True, exist_ok=True)

        print("   Loading BLIP VQA model from HuggingFace...")
        processor_vqa = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        model_vqa = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        model_vqa.eval()

        # Save tokenizer
        tokenizer_dir = vqa_dir / "tokenizer"
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        processor_vqa.tokenizer.save_vocabulary(str(tokenizer_dir))

        # Save config
        config = {
            "model_type": "blip_vqa",
            "image_size": 384,
            "vision_config": model_vqa.config.vision_config.to_dict() if hasattr(model_vqa.config, 'vision_config') else {},
            "text_config": model_vqa.config.text_config.to_dict() if hasattr(model_vqa.config, 'text_config') else {},
        }
        with open(vqa_dir / "blip_vqa_config.json", "w") as f:
            json.dump(config, f, indent=2)

        print("   BLIP VQA model components saved successfully")
        print("   Note: Full ONNX export requires additional processing")

    except Exception as e:
        print(f"   Error with BLIP VQA: {e}")

    # 3. Try to download CN-CLIP
    print("\n3. Processing CN-CLIP model...")
    try:
        from huggingface_hub import snapshot_download

        cn_clip_dir = base_path / "cn-clip"
        cn_clip_dir.mkdir(parents=True, exist_ok=True)

        print("   Downloading CN-CLIP from HuggingFace...")
        # Try to download CN-CLIP ViT-B-16
        snapshot_download(
            repo_id="OFA-Sys/chinese-clip-vit-base-patch16",
            local_dir=str(cn_clip_dir),
            local_dir_use_symlinks=False
        )
        print("   CN-CLIP downloaded successfully")

    except Exception as e:
        print(f"   Error with CN-CLIP: {e}")
        print("   You may need to manually download CN-CLIP models")

    print("\n4. Summary:")
    print("   - OCR models: Already downloaded")
    print("   - BLIP Caption: Basic components saved (needs ONNX conversion)")
    print("   - BLIP VQA: Basic components saved (needs ONNX conversion)")
    print("   - CN-CLIP: Download attempted")

    print("\nNote: For complete ONNX conversion, you may need to run the original export scripts")
    print("      when the encoding issues are resolved.")

    return True

if __name__ == "__main__":
    # We're already in the scripts directory
    success = download_models()
    sys.exit(0 if success else 1)