#!/usr/bin/env python3
"""
动态量化 eisneim CN-CLIP ONNX (vit-b-16) 到 INT8。
输入：
  assets/models/cn-clip-eisneim/vit-b-16.img.fp32.onnx
  assets/models/cn-clip-eisneim/vit-b-16.txt.fp32.onnx
输出：
  assets/models/cn-clip-eisneim/int8/vit-b-16.img.int8.onnx
  assets/models/cn-clip-eisneim/int8/vit-b-16.txt.int8.onnx
"""

from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Quantizing {src.name} -> {dst}")
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        per_channel=True,
        reduce_range=False,
        weight_type=QuantType.QInt8,
    )


def main():
    base = Path(__file__).resolve().parent.parent / "assets" / "models" / "cn-clip-eisneim"
    src_img = base / "vit-b-16.img.fp32.onnx"
    src_txt = base / "vit-b-16.txt.fp32.onnx"
    out_dir = base / "int8"
    out_img = out_dir / "vit-b-16.img.int8.onnx"
    out_txt = out_dir / "vit-b-16.txt.int8.onnx"

    for p in (src_img, src_txt):
        if not p.exists():
            raise FileNotFoundError(f"Missing source model: {p}")

    quantize_file(src_img, out_img)
    quantize_file(src_txt, out_txt)
    print("Done. INT8 models saved to", out_dir)


if __name__ == "__main__":
    main()
