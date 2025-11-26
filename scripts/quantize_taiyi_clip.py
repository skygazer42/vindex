#!/usr/bin/env python3
"""
动态量化 Taiyi CLIP ONNX 到 INT8。
假设已有导出的 Taiyi ONNX：
  assets/models/taiyi-clip/clip_visual.onnx
  assets/models/taiyi-clip/clip_text.onnx
输出：
  assets/models/taiyi-clip/int8/clip_visual.int8.onnx
  assets/models/taiyi-clip/int8/clip_text.int8.onnx
如命名不同，请调整 src_* 路径。
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
    base = Path(__file__).resolve().parent.parent / "assets" / "models" / "taiyi-clip"
    src_img = base / "clip_visual.onnx"
    src_txt = base / "clip_text.onnx"
    out_dir = base / "int8"
    out_img = out_dir / "clip_visual.int8.onnx"
    out_txt = out_dir / "clip_text.int8.onnx"

    for p in (src_img, src_txt):
        if not p.exists():
            raise FileNotFoundError(f"Missing source model: {p}")

    quantize_file(src_img, out_img)
    quantize_file(src_txt, out_txt)
    print("Done. INT8 models saved to", out_dir)


if __name__ == "__main__":
    main()
