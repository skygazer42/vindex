# OCR 文字识别功能设置指南

## 功能概述

VIndex 现已集成 PP-OCRv4 中文文字识别功能，支持：
- 自动检测图像中的文字区域
- 识别中英文混合文字
- 支持多行文本识别
- 可复制识别结果到剪贴板

## 模型下载

OCR 功能需要下载 PP-OCRv4 模型文件。请按以下步骤操作：

### 方法一：使用自动下载脚本（推荐）

1. 打开终端，进入项目的 scripts 目录：
```bash
cd vindex/scripts
```

2. 安装 Python 依赖（如果还没有安装）：
```bash
pip install -r requirements.txt
```

3. 运行 OCR 模型下载脚本：
```bash
python download_ocr_models.py --output ../assets/models/ocr
```

脚本会自动下载以下文件：
- `ch_PP-OCRv4_det_infer.onnx` - 文字检测模型 (~4.5MB)
- `ch_PP-OCRv4_rec_infer.onnx` - 文字识别模型 (~12MB)
- `ppocr_keys_v1.txt` - 字符字典文件
- `ocr_config.json` - 配置文件

### 方法二：手动下载

如果自动下载失败，可以手动下载：

1. 创建目录：
```bash
mkdir -p assets/models/ocr
```

2. 下载模型文件：
- 检测模型：[ch_PP-OCRv4_det_infer.onnx](https://huggingface.co/SWHL/RapidOCR/blob/main/PP-OCRv4/ch_PP-OCRv4_det_infer.onnx)
- 识别模型：[ch_PP-OCRv4_rec_infer.onnx](https://huggingface.co/SWHL/RapidOCR/blob/main/PP-OCRv4/ch_PP-OCRv4_rec_infer.onnx)
- 字典文件：[ppocr_keys_v1.txt](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/ppocr_keys_v1.txt)

3. 将下载的文件放入 `assets/models/ocr/` 目录

4. 创建配置文件 `assets/models/ocr/ocr_config.json`：
```json
{
  "model_type": "pp-ocrv4",
  "det_model": "ch_PP-OCRv4_det_infer.onnx",
  "rec_model": "ch_PP-OCRv4_rec_infer.onnx",
  "dict_file": "ppocr_keys_v1.txt",
  "det_db_thresh": 0.3,
  "det_db_box_thresh": 0.5,
  "det_db_unclip_ratio": 1.6,
  "rec_img_height": 48,
  "rec_img_width": 320,
  "max_side_len": 960,
  "use_angle_cls": false
}
```

## 验证模型文件

确保以下文件存在：
```
assets/models/ocr/
├── ch_PP-OCRv4_det_infer.onnx      # 文字检测模型
├── ch_PP-OCRv4_rec_infer.onnx      # 文字识别模型
├── ppocr_keys_v1.txt               # 字符字典
└── ocr_config.json                 # 配置文件
```

## 编译项目

模型下载完成后，重新编译项目：

```bash
cd vindex
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)  # Linux/Mac
# 或
cmake --build . --config Release  # Windows
```

## 使用 OCR 功能

1. 启动 VIndex 应用
2. 点击 "OCR" 标签页
3. 点击 "Select Image" 选择要识别的图像
4. 点击 "Recognize" 开始识别
5. 识别结果会显示在右侧文本框中
6. 点击 "Copy" 可以复制识别结果到剪贴板

## 支持的图像格式

- PNG
- JPEG/JPG
- BMP
- TIFF
- WebP

## 性能参考

在 Intel i7-12700 CPU 上的典型性能：
- 检测阶段：~100-200ms
- 识别阶段：~50-100ms（每个文本区域）
- 总耗时：取决于图像中的文字数量，通常在 200-500ms

## 常见问题

### Q: 提示 "OCR model not loaded"
**A:** 请检查模型文件是否正确放置在 `assets/models/ocr/` 目录下

### Q: 识别精度不高
**A:**
- 确保图像清晰，文字区域对比度足够
- 尽量使用高分辨率图像
- 避免过度倾斜或扭曲的文字

### Q: 只能识别中文吗？
**A:** PP-OCRv4 模型支持中英文混合识别，也能识别数字和常见标点符号

### Q: 如何调整识别参数？
**A:** 可以修改 `ocr_config.json` 中的参数：
- `det_db_thresh`: 检测阈值（降低可检测更多文字，但可能有误检）
- `det_db_box_thresh`: 文字框阈值
- `max_side_len`: 图像最大边长（影响处理速度和精度）

## 技术细节

OCR 功��采用两阶段处理：
1. **检测阶段**：使用 DBNet 检测文字区域
2. **识别阶段**：使用 CRNN 识别每个文字区域的内容

模型来源：
- 基于 PaddleOCR PP-OCRv4 模型
- 已转换为 ONNX 格式以支持跨平台部署
- 使用 ONNX Runtime 进行推理

## 更新日志

### 2024-11-27
- 完成 OCR 功能集成
- 添加 OCR widget 到主界面
- 支持中英文混合识别
- 添加结果复制功能

## 相关链接

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [RapidOCR](https://github.com/RapidAI/RapidOCR)
- [ONNX Runtime](https://onnxruntime.ai/)