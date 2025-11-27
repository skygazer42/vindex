#!/usr/bin/env python3
"""
Download and export all models for VIndex
Handles encoding issues on Windows
"""

import os
import sys
import subprocess
from pathlib import Path
import json

# Fix encoding issues on Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

def print_status(message, status="INFO"):
    """Print status message without unicode issues"""
    symbols = {
        "INFO": "[i]",
        "SUCCESS": "[+]",
        "ERROR": "[!]",
        "WORKING": "[*]"
    }
    print(f"{symbols.get(status, '[?]')} {message}")

def run_command(cmd, description):
    """Run a command and handle errors"""
    print_status(f"Running: {description}", "WORKING")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        if result.returncode == 0:
            print_status(f"Success: {description}", "SUCCESS")
            return True
        else:
            print_status(f"Failed: {description}", "ERROR")
            if result.stderr:
                print(f"  Error: {result.stderr[:200]}")
            return False
    except Exception as e:
        print_status(f"Exception in {description}: {str(e)}", "ERROR")
        return False

def download_ocr_models():
    """Download OCR models - Already done but checking"""
    print_status("Checking OCR models...", "INFO")
    ocr_dir = Path("../assets/models/ocr")

    if ocr_dir.exists() and len(list(ocr_dir.glob("*.onnx"))) >= 2:
        print_status("OCR models already downloaded", "SUCCESS")
        return True

    # Run the OCR download script
    return run_command(
        "python download_ocr_models.py --output ../assets/models/ocr",
        "Download OCR models"
    )

def download_and_export_clip():
    """Download and export CLIP models"""
    print_status("Processing CLIP models...", "INFO")

    # First, try to download using simplified approach
    print_status("Downloading CLIP models from HuggingFace...", "WORKING")

    try:
        from huggingface_hub import snapshot_download, hf_hub_download
        import torch
        import onnx

        # Download CN-CLIP
        try:
            print_status("Downloading CN-CLIP ViT-B/16...", "WORKING")

            # Option 1: Try official CN-CLIP
            model_dir = Path("../assets/models/cn-clip")
            model_dir.mkdir(parents=True, exist_ok=True)

            # Download model files
            snapshot_download(
                repo_id="OFA-Sys/chinese-clip-vit-base-patch16",
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
                ignore_patterns=["*.bin", "*.safetensors"]  # Skip large pytorch files
            )
            print_status("CN-CLIP config downloaded", "SUCCESS")

        except Exception as e:
            print_status(f"CN-CLIP download error: {str(e)[:100]}", "ERROR")

        # Try to export using OpenCLIP
        try:
            print_status("Trying OpenCLIP export...", "WORKING")
            import open_clip

            # Load model
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-B-16',
                pretrained='openai'
            )
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model.eval()

            # Export visual encoder
            print_status("Exporting visual encoder...", "WORKING")
            dummy_image = torch.randn(1, 3, 224, 224)

            visual_path = Path("../assets/models/clip_visual.onnx")
            if not visual_path.exists():
                torch.onnx.export(
                    model.visual,
                    dummy_image,
                    str(visual_path),
                    input_names=['image'],
                    output_names=['features'],
                    dynamic_axes={'image': {0: 'batch_size'}},
                    opset_version=14
                )
                print_status("Visual encoder exported", "SUCCESS")
            else:
                print_status("Visual encoder already exists", "INFO")

            # Export text encoder
            print_status("Exporting text encoder...", "WORKING")
            dummy_text = torch.randint(0, 1000, (1, 77))  # CLIP uses 77 token context

            text_path = Path("../assets/models/clip_text.onnx")
            if not text_path.exists():
                torch.onnx.export(
                    model.text,
                    dummy_text,
                    str(text_path),
                    input_names=['text'],
                    output_names=['features'],
                    dynamic_axes={'text': {0: 'batch_size'}},
                    opset_version=14
                )
                print_status("Text encoder exported", "SUCCESS")
            else:
                print_status("Text encoder already exists", "INFO")

        except Exception as e:
            print_status(f"OpenCLIP export error: {str(e)[:100]}", "ERROR")

    except ImportError as e:
        print_status(f"Import error: {str(e)}", "ERROR")
        print_status("Installing required packages...", "WORKING")
        run_command("pip install open-clip-torch torch transformers huggingface_hub", "Install packages")

    return True

def download_and_export_blip():
    """Download and export BLIP models"""
    print_status("Processing BLIP Caption models...", "INFO")

    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        import torch

        blip_dir = Path("../assets/models/blip")
        blip_dir.mkdir(parents=True, exist_ok=True)

        # Check if we need to download
        if not (blip_dir / "pytorch_model.bin").exists():
            print_status("Downloading BLIP from Salesforce...", "WORKING")

            processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                cache_dir=str(blip_dir / "cache")
            )
            model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                cache_dir=str(blip_dir / "cache")
            )

            # Save processor and model
            processor.save_pretrained(str(blip_dir))
            model.save_pretrained(str(blip_dir))

            print_status("BLIP Caption model downloaded", "SUCCESS")
        else:
            print_status("BLIP Caption model already exists", "INFO")

        # Try to export to ONNX (simplified version)
        # Note: Full export requires complex handling of encoder-decoder architecture
        print_status("BLIP ONNX export requires manual conversion", "INFO")

    except Exception as e:
        print_status(f"BLIP Caption error: {str(e)[:100]}", "ERROR")

    return True

def download_and_export_blip_vqa():
    """Download and export BLIP VQA models"""
    print_status("Processing BLIP VQA models...", "INFO")

    try:
        from transformers import BlipProcessor, BlipForQuestionAnswering
        import torch

        vqa_dir = Path("../assets/models/blip_vqa")
        vqa_dir.mkdir(parents=True, exist_ok=True)

        # Check if we need to download
        if not (vqa_dir / "pytorch_model.bin").exists():
            print_status("Downloading BLIP VQA from Salesforce...", "WORKING")

            processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-vqa-base",
                cache_dir=str(vqa_dir / "cache")
            )
            model = BlipForQuestionAnswering.from_pretrained(
                "Salesforce/blip-vqa-base",
                cache_dir=str(vqa_dir / "cache")
            )

            # Save processor and model
            processor.save_pretrained(str(vqa_dir))
            model.save_pretrained(str(vqa_dir))

            print_status("BLIP VQA model downloaded", "SUCCESS")
        else:
            print_status("BLIP VQA model already exists", "INFO")

        # Try to export to ONNX (simplified version)
        print_status("BLIP VQA ONNX export requires manual conversion", "INFO")

    except Exception as e:
        print_status(f"BLIP VQA error: {str(e)[:100]}", "ERROR")

    return True

def download_chinese_clip():
    """Download Chinese CLIP models"""
    print_status("Processing Chinese CLIP models...", "INFO")

    try:
        from huggingface_hub import snapshot_download

        # Download CN-CLIP
        cn_clip_dir = Path("../assets/models/cn-clip-full")
        cn_clip_dir.mkdir(parents=True, exist_ok=True)

        print_status("Downloading full CN-CLIP model...", "WORKING")
        snapshot_download(
            repo_id="OFA-Sys/chinese-clip-vit-base-patch16",
            local_dir=str(cn_clip_dir),
            local_dir_use_symlinks=False
        )
        print_status("CN-CLIP full model downloaded", "SUCCESS")

        # Download Taiyi-CLIP
        taiyi_dir = Path("../assets/models/taiyi-clip-full")
        taiyi_dir.mkdir(parents=True, exist_ok=True)

        print_status("Downloading Taiyi-CLIP model...", "WORKING")
        snapshot_download(
            repo_id="IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese",
            local_dir=str(taiyi_dir),
            local_dir_use_symlinks=False,
            ignore_patterns=["*.bin"]  # Skip large files if needed
        )
        print_status("Taiyi-CLIP downloaded", "SUCCESS")

    except Exception as e:
        print_status(f"Chinese CLIP error: {str(e)[:100]}", "ERROR")

    return True

def verify_models():
    """Verify which models are present"""
    print_status("\n=== Model Verification ===", "INFO")

    models_dir = Path("../assets/models")

    # Check OCR
    ocr_dir = models_dir / "ocr"
    ocr_files = list(ocr_dir.glob("*.onnx")) if ocr_dir.exists() else []
    if len(ocr_files) >= 2:
        print_status("OCR models: READY", "SUCCESS")
    else:
        print_status("OCR models: MISSING", "ERROR")

    # Check CLIP
    clip_visual = models_dir / "clip_visual.onnx"
    clip_text = models_dir / "clip_text.onnx"
    if clip_visual.exists() and clip_text.exists():
        print_status("CLIP ONNX models: READY", "SUCCESS")
    else:
        print_status("CLIP ONNX models: MISSING", "ERROR")

    # Check BLIP
    blip_dir = models_dir / "blip"
    if blip_dir.exists() and list(blip_dir.glob("*.json")):
        print_status("BLIP Caption: PARTIALLY READY (needs ONNX)", "INFO")
    else:
        print_status("BLIP Caption: MISSING", "ERROR")

    # Check BLIP VQA
    vqa_dir = models_dir / "blip_vqa"
    if vqa_dir.exists() and list(vqa_dir.glob("*.json")):
        print_status("BLIP VQA: PARTIALLY READY (needs ONNX)", "INFO")
    else:
        print_status("BLIP VQA: MISSING", "ERROR")

    # Check Chinese models
    cn_full = models_dir / "cn-clip-full"
    if cn_full.exists() and list(cn_full.glob("*")):
        print_status("CN-CLIP full: DOWNLOADED", "SUCCESS")

    taiyi_full = models_dir / "taiyi-clip-full"
    if taiyi_full.exists() and list(taiyi_full.glob("*")):
        print_status("Taiyi-CLIP: DOWNLOADED", "SUCCESS")

def main():
    print_status("=== VIndex Model Download Tool ===", "INFO")
    print_status("This will download all required models", "INFO")
    print()

    # Change to scripts directory
    os.chdir(Path(__file__).parent)

    # Install basic requirements first
    print_status("Installing/updating required packages...", "WORKING")
    run_command(
        "pip install -q torch torchvision transformers huggingface_hub open-clip-torch onnx",
        "Install dependencies"
    )

    # Download models
    tasks = [
        ("OCR Models", download_ocr_models),
        ("CLIP Models", download_and_export_clip),
        ("BLIP Caption", download_and_export_blip),
        ("BLIP VQA", download_and_export_blip_vqa),
        ("Chinese CLIP", download_chinese_clip),
    ]

    for task_name, task_func in tasks:
        print()
        print_status(f"=== {task_name} ===", "INFO")
        try:
            task_func()
        except Exception as e:
            print_status(f"Task failed: {str(e)[:200]}", "ERROR")

    # Verify
    print()
    verify_models()

    print()
    print_status("=== Download Complete ===", "INFO")
    print_status("Check the verification results above", "INFO")
    print_status("Some models may need manual ONNX conversion", "INFO")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()