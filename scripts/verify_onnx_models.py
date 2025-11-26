#!/usr/bin/env python3
"""
验证下载的ONNX模型
测试CN-CLIP (eisneim) 的ONNX模型是否可以正常加载和推理
"""

import sys
from pathlib import Path
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("❌ onnxruntime未安装")
    print("   安装: pip install onnxruntime")
    sys.exit(1)


def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def verify_model(model_path, input_shape=None, input_name="input"):
    """验证ONNX模型"""
    try:
        # 加载模型
        print(f"\n📥 加载模型: {model_path.name}")
        sess = ort.InferenceSession(str(model_path))

        # 打印模型信息
        inputs = sess.get_inputs()
        outputs = sess.get_outputs()

        print(f"   输入数量: {len(inputs)}")
        for inp in inputs:
            print(f"     - {inp.name}: {inp.shape} ({inp.type})")

        print(f"   输出数量: {len(outputs)}")
        for out in outputs:
            print(f"     - {out.name}: {out.shape} ({out.type})")

        # 如果未提供input_shape，从模型输入自动推断
        if input_shape is None:
            # 获取模型期望的输入形状
            input_shape_from_model = inputs[0].shape
            # 替换动态维度（如batch_size）为1
            input_shape = tuple(1 if isinstance(dim, str) else dim for dim in input_shape_from_model)
            print(f"   自动推断输入形状: {input_shape}")

        # 创建测试输入
        if inputs[0].type == 'tensor(float)':
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
        else:
            dummy_input = np.random.randint(0, 21128, input_shape).astype(np.int64)

        # 推理测试
        print(f"   🔄 测试推理...")
        input_dict = {inputs[0].name: dummy_input}
        result = sess.run(None, input_dict)

        print(f"   ✅ 推理成功!")
        print(f"   输出形状: {result[0].shape}")
        print(f"   输出范围: [{result[0].min():.4f}, {result[0].max():.4f}]")

        return True, sess

    except Exception as e:
        print(f"   ❌ 失败: {e}")
        return False, None


def verify_cn_clip_onnx():
    """验证CN-CLIP ONNX模型"""
    print_header("🌏 验证 CN-CLIP ONNX 模型")

    base_path = Path("../assets/models/cn-clip-eisneim")

    if not base_path.exists():
        print(f"❌ 模型目录不存在: {base_path}")
        return False

    print(f"📂 模型目录: {base_path.absolute()}")

    results = {}

    # 测试图像编码器 (FP32)
    print("\n" + "-" * 60)
    print("测试 1: 图像编码器 (FP32)")
    print("-" * 60)
    img_fp32_path = base_path / "vit-b-16.img.fp32.onnx"
    if img_fp32_path.exists():
        success, sess = verify_model(img_fp32_path, (1, 3, 224, 224))
        results['img_fp32'] = success
    else:
        print(f"⚠️  文件不存在: {img_fp32_path}")
        results['img_fp32'] = False

    # 测试文本编码器 (FP32)
    print("\n" + "-" * 60)
    print("测试 2: 文本编码器 (FP32)")
    print("-" * 60)
    txt_fp32_path = base_path / "vit-b-16.txt.fp32.onnx"
    if txt_fp32_path.exists():
        # 自动推断正确的输入形状
        success, sess = verify_model(txt_fp32_path, input_shape=None)
        results['txt_fp32'] = success
    else:
        print(f"⚠️  文件不存在: {txt_fp32_path}")
        results['txt_fp32'] = False

    # 测试图像编码器 (FP16)
    print("\n" + "-" * 60)
    print("测试 3: 图像编码器 (FP16)")
    print("-" * 60)
    img_fp16_path = base_path / "vit-b-16.img.fp16.onnx"
    if img_fp16_path.exists():
        success, sess = verify_model(img_fp16_path, (1, 3, 224, 224))
        results['img_fp16'] = success
    else:
        print(f"⚠️  文件不存在: {img_fp16_path}")
        results['img_fp16'] = False

    # 测试文本编码器 (FP16)
    print("\n" + "-" * 60)
    print("测试 4: 文本编码器 (FP16)")
    print("-" * 60)
    txt_fp16_path = base_path / "vit-b-16.txt.fp16.onnx"
    if txt_fp16_path.exists():
        # 自动推断正确的输入形状
        success, sess = verify_model(txt_fp16_path, input_shape=None)
        results['txt_fp16'] = success
    else:
        print(f"⚠️  文件不存在: {txt_fp16_path}")
        results['txt_fp16'] = False

    return results


def print_summary(results):
    """打印测试总结"""
    print_header("📊 测试总结")

    total = len(results)
    passed = sum(1 for v in results.values() if v)

    print(f"\n总计: {total} 个模型")
    print(f"通过: {passed} 个 ✅")
    print(f"失败: {total - passed} 个 ❌")
    print()

    for name, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {name}")

    if passed == total:
        print("\n🎉 所有模型验证通过！可以开始C++集成。")
        return True
    else:
        print("\n⚠️  部分模型验证失败，请检查。")
        return False


def main():
    print("🔍 ONNX模型验证工具")
    print()

    # 验证CN-CLIP ONNX模型
    results = verify_cn_clip_onnx()

    # 打印总结
    all_passed = print_summary(results)

    # 推荐下一步
    if all_passed:
        print_header("🚀 下一步")
        print("""
推荐使用FP32版本进行开发（精度最高）：

C++代码示例：
    ChineseClipEncoder encoder(
        "assets/models/cn-clip-eisneim/vit-b-16.img.fp32.onnx",
        "assets/models/cn-clip-eisneim/vit-b-16.txt.fp32.onnx",
        "assets/models/cn-clip/vocab.txt",
        512
    );

如需更快速度，可使用FP16版本（精度略降<1%）：
    ChineseClipEncoder encoder(
        "assets/models/cn-clip-eisneim/vit-b-16.img.fp16.onnx",
        "assets/models/cn-clip-eisneim/vit-b-16.txt.fp16.onnx",
        "assets/models/cn-clip/vocab.txt",
        512
    );
        """)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
