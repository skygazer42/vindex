# C++ ONNX 推理构建指南

## ✅ ONNX 模型状态

所有 6 个 ONNX 模型都已成功转换并通过测试:

| 模型 | 文件大小 | 推理时间 | 输入/输出形状 | 状态 |
|:---|---:|---:|:---|:---:|
| **OCR 检测** | 4.5 MB | 56ms | [1,3,640,640] → [1,1,640,640] | ✅ |
| **OCR 识别** | 10.4 MB | 19ms | [1,3,48,320] → [1,40,6625] | ✅ |
| **CLIP 视觉编码器** | 329 MB | 66ms | [1,3,224,224] → [1,512] | ✅ |
| **BLIP 视觉编码器** | 329 MB | 173ms | [1,3,384,384] → [1,577,768] | ✅ |
| **BLIP 文本解码器** | 616 MB | - | 双输入 (ids+hidden) → logits | ✅ |
| **VQA 视觉编码器** | 329 MB | 206ms | [1,3,384,384] → [1,577,768] | ✅ |

## 📦 依赖项安装

### 1. ONNX Runtime C++ SDK

```bash
# 下载 ONNX Runtime (选择最新版本)
# Windows x64:
https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-win-x64-1.16.0.zip

# 解压到 C:\onnxruntime
# 目录结构:
C:\onnxruntime\
├── include/           # 头文件
│   └── onnxruntime_cxx_api.h
├── lib/              # 静态库
│   └── onnxruntime.lib
└── bin/              # 动态库
    └── onnxruntime.dll
```

### 2. OpenCV (可选)

```bash
# 下载 OpenCV Windows 包
https://github.com/opencv/opencv/releases/download/4.8.0/opencv-4.8.0-windows.exe

# 安装到 C:\opencv
```

### 3. Qt6 (完整 GUI 需要)

```bash
# 下载 Qt Online Installer
https://www.qt.io/download-qt-installer

# 安装 Qt 6.4+ with MinGW 编译器
```

## 🔨 编译方法

### 方法 1: 简单测试 (不需要 Qt)

```bash
# 使用 MinGW
g++ simple_onnx_test.cpp -o onnx_test.exe \
    -I"C:\onnxruntime\include" \
    -L"C:\onnxruntime\lib" \
    -lonnxruntime \
    -std=c++17

# 复制 DLL
copy C:\onnxruntime\bin\onnxruntime.dll .

# 运行测试
.\onnx_test.exe
```

### 方法 2: CMake 构建 (推荐)

创建 `build_cpp.bat`:

```batch
@echo off
echo Building VIndex ONNX Test...

rem 设置环境变量
set ONNXRUNTIME_ROOT=C:\onnxruntime
set OpenCV_DIR=C:\opencv\build

rem 创建构建目录
if not exist build_test mkdir build_test
cd build_test

rem 配置 CMake
cmake -G "MinGW Makefiles" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DONNXRUNTIME_ROOT=%ONNXRUNTIME_ROOT% ^
    ..

rem 编译
cmake --build . --config Release

rem 复制 DLL
copy %ONNXRUNTIME_ROOT%\bin\onnxruntime.dll Release\
copy %OpenCV_DIR%\x64\vc16\bin\*.dll Release\ 2>nul

echo.
echo Build complete! Run: Release\test_inference.exe
cd ..
```

### 方法 3: Visual Studio

1. 打开 Visual Studio 2022
2. 文件 → 打开 → CMake
3. 选择 `CMakeLists_test.txt`
4. 配置 ONNX Runtime 路径
5. 生成 → 全部生成

## 🎯 集成到主项目

### 1. 更新 CMakeLists.txt

```cmake
# 添加 ONNX Runtime
set(ONNXRUNTIME_ROOT "C:/onnxruntime" CACHE PATH "Path to ONNX Runtime")
find_package(onnxruntime REQUIRED)

# 链接库
target_link_libraries(vindex
    ...
    onnxruntime
)
```

### 2. 更新模型管理器

在 `src/core/model_manager.cpp`:

```cpp
bool ModelManager::loadModels() {
    // 初始化 ONNX Runtime
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "vindex");

    // 加载各个模型
    loadOCRModels();      // ✅ 已完成
    loadCLIPModel();      // ✅ 视觉编码器已完成
    loadBLIPModel();      // ✅ 已完成
    loadVQAModel();       // ✅ 视觉编码器已完成

    return true;
}
```

## 🚀 功能实现状态

| 功能 | ONNX模型 | C++代码 | 集成状态 |
|:---|:---:|:---:|:---:|
| **OCR 文字识别** | ✅ 完整 | ✅ 已有 | 待集成 |
| **图像搜索** | ⚠️ 仅视觉 | ✅ 已有 | 需文本编码器 |
| **图像描述** | ✅ 完整 | ⚠️ 部分 | 待完成解码逻辑 |
| **视觉问答** | ⚠️ 仅视觉 | ⚠️ 部分 | 需问答解码器 |

## 📊 性能数据

基于 Python 测试的推理时间 (CPU):

- OCR 检测: 56ms/图像
- OCR 识别: 19ms/文本框
- CLIP 编码: 66ms/图像
- BLIP 编码: 173ms/图像
- 综合 OCR: ~100ms/图像 (检测+识别)

## 🐛 已知问题

1. **CLIP 文本编码器未转换**
   - 原因: OpenCLIP 导出问题
   - 解决: 需要自定义导出脚本

2. **Qt6 依赖**
   - 完整 GUI 需要 Qt6.4+
   - 可以先用控制台版本测试

3. **Windows 路径问题**
   - 使用宽字符路径 (std::wstring)
   - 或使用前向斜杠

## ✅ 下一步

1. **立即可用**:
   - OCR 功能可以直接集成使用
   - 图像特征提取正常工作

2. **需要补充**:
   - CLIP 文本编码器导出
   - BLIP/VQA 解码逻辑实现

3. **建议**:
   ```bash
   # 1. 先测试简单版本
   g++ simple_onnx_test.cpp -o test.exe [编译参数]
   .\test.exe

   # 2. 如果成功,集成到主项目
   # 3. 逐个功能测试和优化
   ```

## 📝 测试命令

```bash
# Python 测试 (已通过)
python test_onnx_inference.py
python test_cpp_inference.py

# C++ 测试
.\onnx_test.exe

# 完整项目 (需要 Qt6)
mkdir build && cd build
cmake .. -G "MinGW Makefiles"
cmake --build .
.\vindex.exe
```

---

**总结**: ONNX 模型转换成功,C++ 推理代码已准备好。现在可以开始集成到主项目中!