#!/usr/bin/env python3
"""
Test ONNX model inference in Python
"""
import os
import sys
from pathlib import Path
import numpy as np

# Fix encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def test_model(model_path, name, input_shape=None):
    """Test loading and running an ONNX model"""
    try:
        import onnxruntime as ort
    except ImportError:
        print("Installing onnxruntime...")
        os.system("pip install -q onnxruntime")
        import onnxruntime as ort

    print(f"\nTesting {name}...")
    print(f"  Path: {model_path}")

    if not Path(model_path).exists():
        print(f"  [ERROR] Model file not found")
        return False

    try:
        # Create session
        session = ort.InferenceSession(str(model_path))

        # Get input/output info
        inputs = session.get_inputs()
        outputs = session.get_outputs()

        print(f"  Inputs: {len(inputs)}")
        for i, inp in enumerate(inputs):
            print(f"    [{i}] {inp.name}: {inp.shape} ({inp.type})")

        print(f"  Outputs: {len(outputs)}")
        for i, out in enumerate(outputs):
            print(f"    [{i}] {out.name}: {out.shape} ({out.type})")

        # Try inference with dummy data
        if input_shape:
            print(f"  Running inference with shape {input_shape}...")

            # Create dummy input
            if inputs[0].type == 'tensor(float)':
                dummy_input = np.random.randn(*input_shape).astype(np.float32)
            elif inputs[0].type == 'tensor(int64)':
                dummy_input = np.random.randint(0, 100, input_shape).astype(np.int64)
            else:
                print(f"  [WARNING] Unknown input type: {inputs[0].type}")
                return True

            # Run inference
            input_name = inputs[0].name
            outputs = session.run(None, {input_name: dummy_input})

            print(f"  Output shape: {outputs[0].shape}")
            print(f"  Output dtype: {outputs[0].dtype}")

            # Check output
            if len(outputs[0].shape) > 0:
                print(f"  [SUCCESS] Inference completed successfully!")
                return True
            else:
                print(f"  [WARNING] Empty output")
                return False
        else:
            print(f"  [SUCCESS] Model loaded successfully!")
            return True

    except Exception as e:
        print(f"  [ERROR] {str(e)[:200]}")
        return False

def main():
    print("=== ONNX Model Inference Test ===")

    models_dir = Path("assets/models")

    test_cases = [
        # Model path, Name, Input shape (optional)
        (models_dir / "ocr" / "ch_PP-OCRv4_det_infer.onnx", "OCR Detection", (1, 3, 640, 640)),
        (models_dir / "ocr" / "ch_PP-OCRv4_rec_infer.onnx", "OCR Recognition", (1, 3, 48, 320)),
        (models_dir / "clip_visual.onnx", "CLIP Visual Encoder", (1, 3, 224, 224)),
        (models_dir / "blip" / "blip_visual_encoder.onnx", "BLIP Visual Encoder", (1, 3, 384, 384)),
        (models_dir / "blip" / "blip_text_decoder.onnx", "BLIP Text Decoder", None),  # Complex inputs
        (models_dir / "blip_vqa" / "blip_vqa_visual_encoder.onnx", "VQA Visual Encoder", (1, 3, 384, 384)),
    ]

    results = {}
    for model_path, name, input_shape in test_cases:
        success = test_model(model_path, name, input_shape)
        results[name] = success

    # Summary
    print("\n=== Summary ===")
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)

    for name, success in results.items():
        status = "[OK]" if success else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\nPassed: {success_count}/{total_count}")

    if success_count == total_count:
        print("\nAll models working correctly! Ready for C++ integration.")
    elif success_count > 0:
        print(f"\n{success_count} models working. Some issues need to be resolved.")
    else:
        print("\nNo models working. Please check the model files.")

    return success_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)