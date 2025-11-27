<p align="center">
  <img src="docs/images/logo.png" alt="VIndex Logo" width="120" height="120">
</p>

<h1 align="center">VIndex</h1>

<p align="center">
  <strong>Visual Search Engine - 视觉搜索引擎</strong>
</p>

<p align="center">
  基于 CN-CLIP + FAISS 的跨平台图像检索与理解应用
</p>

<p align="center">
  <img src="https://img.shields.io/badge/C++-17-blue.svg" alt="C++17">
  <img src="https://img.shields.io/badge/Qt-6.6-41CD52?logo=qt&logoColor=white" alt="Qt6">
  <img src="https://img.shields.io/badge/OpenCV-4.10-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/ONNX_Runtime-1.16-FF6F00?logo=onnx&logoColor=white" alt="ONNX Runtime">
  <img src="https://img.shields.io/badge/FAISS-1.7-0467DF?logo=meta&logoColor=white" alt="FAISS">
  <img src="https://img.shields.io/badge/CN--CLIP-ViT--B/16-FF4500" alt="CN-CLIP">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg" alt="Platform">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License">
</p>

<p align="center">
  <a href="#功能特性">功能特性</a> •
  <a href="#快速开始">快速开始</a> •
  <a href="#安装指南">安装指南</a> •
  <a href="#使用说明">使用说明</a> •
  <a href="#文档">文档</a>
</p>

---

## 功能特性

| 功能 | 描述 | 模型 |
|:---:|:---|:---|
| **以图搜图** | 上传图片，检索相似图像 | CN-CLIP ViT-B/16 |
| **以文搜图** | 输入中/英文描述，搜索匹配图像 | CN-CLIP Text Encoder |
| **图文匹配** | 计算图像与文本的相似度得分 | CN-CLIP 双编码器 |
| **图像描述** | 自动生成图像的文字描述 | BLIP Caption |
| **视觉问答** | 针对图像内容进行问答 | BLIP VQA |
| **图库管理** | 导入、浏览、分类、删除图像 | SQLite + FAISS |

### 亮点

- **中文优化** - 采用 CN-CLIP 模型，中文搜索效果显著优于原版 CLIP
- **高性能检索** - FAISS 向量索引，百万级图库毫秒级响应
- **跨平台支持** - Windows / Linux / macOS 原生运行
- **中英双语界面** - 支持运行时切换语言
- **现代化 UI** - 基于 Qt6 的美观界面，支持自定义主题

---

## 界面预览

<p align="center">
  <img src="docs/images/screenshot_image_search.png" alt="Image Search" width="45%">
  <img src="docs/images/screenshot_text_search.png" alt="Text Search" width="45%">
</p>

---

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/your-username/vindex.git
cd vindex
```

### 2. 安装依赖

**Ubuntu/Debian:**
```bash
sudo apt install -y build-essential cmake git \
    qt6-base-dev libopencv-dev libsqlite3-dev libxcb-cursor0
```

**Windows:** 参考 [Windows 完整构建指南](docs/WINDOWS_BUILD_GUIDE.md)

### 3. 编译运行

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./VIndex
```

### 4. 导出模型

```bash
cd scripts
pip install -r requirements.txt
python export_cn_clip_onnx.py --model ViT-B-16 --output ../assets/models
```

---

## 安装指南

### 依赖版本

| 依赖 | 最低版本 | 推荐版本 |
|:---|:---:|:---:|
| CMake | 3.20 | 3.28+ |
| Qt | 6.2 | 6.6.3 |
| OpenCV | 4.5 | 4.10 |
| ONNX Runtime | 1.14 | 1.16+ |
| FAISS | 1.7 | 1.7.4 |
| SQLite | 3.30 | 3.45 |
| C++ 编译器 | C++17 | GCC 11+ / MSVC 2022 |

### 平台安装

<details>
<summary><b>Linux (Ubuntu/Debian)</b></summary>

```bash
# 系统依赖
sudo apt install -y build-essential cmake git \
    qt6-base-dev libopencv-dev libsqlite3-dev libxcb-cursor0

# ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar -xzf onnxruntime-linux-x64-1.16.3.tgz
sudo mv onnxruntime-linux-x64-1.16.3 /usr/local/onnxruntime

# FAISS
git clone https://github.com/facebookresearch/faiss.git
cd faiss && mkdir build && cd build
cmake .. -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF
make -j$(nproc) && sudo make install
```

</details>

<details>
<summary><b>Windows</b></summary>

详细步骤请参考 **[Windows 完整构建指南](docs/WINDOWS_BUILD_GUIDE.md)**

快速概览：
1. 安装 Visual Studio 2022 (C++ 桌面开发)
2. 安装 CMake 3.28+
3. 安装 Qt 6.6.3 (MSVC 2019 64-bit)
4. 下载 OpenCV、ONNX Runtime、FAISS
5. 配置环境变量
6. CMake 构建

</details>

<details>
<summary><b>macOS</b></summary>

```bash
# Homebrew 安装依赖
brew install cmake qt@6 opencv sqlite faiss

# ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-osx-arm64-1.16.3.tgz
tar -xzf onnxruntime-osx-arm64-1.16.3.tgz
export ONNXRUNTIME_ROOT=$PWD/onnxruntime-osx-arm64-1.16.3

# 编译
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DQt6_DIR=$(brew --prefix qt@6)/lib/cmake/Qt6
make -j$(sysctl -n hw.ncpu)
```

</details>

---

## 使用说明

### 功能模块

| 标签页 | 功能 | 使用方法 |
|:---|:---|:---|
| **Image Search** | 以图搜图 | 点击"选择图像"，设置参数后点击"搜索" |
| **Text Search** | 以文搜图 | 输入描述文字（支持中英文），点击"搜索" |
| **Match** | 图文匹配 | 选择图像，输入文本，点击"计算相似度" |
| **Caption** | 图像描述 | 选择图像，点击"生成描述" |
| **VQA** | 视觉问答 | 选择图像，输入问题，点击"提问" |
| **Library** | 图库管理 | 浏览、筛选、删除已导入的图像 |

### 导入图像

1. 菜单 `File → Import Folder...` 或工具栏 `Import Folder`
2. 选择图像文件夹
3. 选择是否包含子目录
4. 等待导入完成

### 切换语言

菜单 `Settings → Language → 中文/English`

### 快捷键

| 快捷键 | 功能 |
|:---:|:---|
| `Ctrl+I` | 导入文件夹 |
| `Ctrl+Q` | 退出程序 |

---

## 项目结构

```
vindex/
├── src/
│   ├── core/                    # 核心模块
│   │   ├── clip_encoder.*       # CLIP 编码器
│   │   ├── model_manager.*      # 模型管理器
│   │   ├── caption_model.*      # 图像描述模型
│   │   └── vqa_model.*          # VQA 模型
│   ├── index/                   # 索引模块
│   │   ├── faiss_index.*        # FAISS 向量索引
│   │   └── database_manager.*   # 数据库管理
│   ├── gui/                     # GUI 模块
│   │   ├── main_window.*        # 主窗口
│   │   ├── image_search_widget.*# 图搜图
│   │   ├── text_search_widget.* # 文搜图
│   │   └── ...                  # 其他功能组件
│   └── utils/                   # 工具模块
│       ├── translator.*         # 多语言支持
│       └── api_client.*         # API 客户端
├── assets/
│   ├── models/                  # ONNX 模型文件
│   └── vocab/                   # 词表文件
├── scripts/                     # Python 工具脚本
│   ├── export_cn_clip_onnx.py   # CN-CLIP 导出
│   └── requirements.txt         # Python 依赖
├── resources/
│   └── styles/                  # QSS 样式表
├── docs/                        # 文档
└── CMakeLists.txt               # CMake 配置
```

---

## 文档

| 文档 | 描述 |
|:---|:---|
| [快速开始指南](docs/QUICKSTART.md) | 5分钟上手教程 |
| [Windows 构建指南](docs/WINDOWS_BUILD_GUIDE.md) | Windows 从零开始完整指南 |
| [编译指南](docs/BUILD.md) | 多平台编译说明 |
| [CN-CLIP 支持](docs/CHINESE_CLIP_SUPPORT.md) | 中文 CLIP 模型配置 |
| [项目技术总结](docs/PROJECT_SUMMARY.md) | 架构设计与实现细节 |

---

## 模型说明

本项目使用多个 AI 模型实现不同的视觉理解功能：

### CN-CLIP (Chinese-CLIP)

| 组件 | 文件 | 作用 |
|:---|:---|:---|
| **视觉编码器** | `clip_visual.onnx` | 将图像编码为 512 维特征向量，用于图像检索和匹配 |
| **文本编码器** | `clip_text.onnx` | 将中/英文文本编码为 512 维特征向量，用于文本搜索 |
| **词表** | `clip_vocab.txt` | BERT 中文词表 (21128 词)，用于文本分词 |

**模型来源**: [OFA-Sys/Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP)

**特点**:
- 基于 OpenAI CLIP 架构，针对中文优化
- 图像和文本共享语义空间，支持跨模态检索
- ViT-B/16 版本平衡了性能和效率

**应用场景**:
- 以图搜图：提取查询图像特征 → FAISS 检索相似向量
- 以文搜图：提取文本特征 → FAISS 检索匹配图像
- 图文匹配：计算图像和文本特征的余弦相似度

### Taiyi-BLIP (太乙-BLIP)

| 组件 | 文件 | 作用 |
|:---|:---|:---|
| **视觉编码器** | `blip_visual_encoder.onnx` | 提取图像视觉特征，输出 patch embeddings |
| **文本解码器** | `blip_text_decoder.onnx` | 自回归生成图像描述文本 |
| **配置文件** | `blip_config.json` | 模型参数配置（图像尺寸、词表大小等） |
| **词表** | `tokenizer/vocab.txt` | BERT 中文词表，用于文本生成和解码 |

**模型来源**: [IDEA-CCNL/Taiyi-BLIP-750M-Chinese](https://huggingface.co/IDEA-CCNL/Taiyi-BLIP-750M-Chinese)

**特点**:
- 基于 Salesforce BLIP 架构，750M 参数
- 专门针对中文图像描述任务微调
- 支持生成流畅的中文图像描述

**应用场景**:
- 图像描述：输入图像 → 视觉编码 → 文本解码 → 中文描述

### 模型文件结构

```
assets/models/
├── clip_visual.onnx          # CN-CLIP 视觉编码器 (~350MB)
├── clip_text.onnx            # CN-CLIP 文本编码器 (~250MB)
├── blip/                     # BLIP 模型目录
│   ├── blip_visual_encoder.onnx   # BLIP 视觉编码器 (~400MB)
│   ├── blip_text_decoder.onnx     # BLIP 文本解码器 (~450MB)
│   ├── blip_config.json           # 模型配置
│   └── tokenizer/
│       └── vocab.txt              # 词表文件
assets/vocab/
└── clip_vocab.txt            # CN-CLIP 词表
```

### 导出模型

```bash
cd scripts

# 导出 CN-CLIP 模型 (用于搜索和匹配)
python export_cn_clip_onnx.py --model ViT-B-16 --output ../assets/models

# 导出 BLIP 模型 (用于图像描述)
python export_blip_onnx.py --output ../assets/models/blip
```

---

## 数据流

```
┌─────────────────────────────────────────────────────────────────────┐
│                           VIndex 数据流                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  以图搜图:                                                           │
│  Image → OpenCV预处理 → CLIP Visual → 512D向量 → FAISS检索 → 结果    │
│                                                                     │
│  以文搜图:                                                           │
│  Text → Tokenizer → CLIP Text → 512D向量 → FAISS检索 → 结果          │
│                                                                     │
│  图文匹配:                                                           │
│  Image + Text → CLIP双编码器 → 余弦相似度 → 得分                      │
│                                                                     │
│  图像描述:                                                           │
│  Image → BLIP Encoder → Decoder → Caption                           │
│                                                                     │
│  视觉问答:                                                           │
│  Image + Question → BLIP VQA → Answer                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 性能参考

| 操作 | 数据量 | 耗时 | 环境 |
|:---|:---:|:---:|:---|
| 图像编码 | 1张 | ~50ms | CPU (i7-12700) |
| 文本编码 | 1条 | ~10ms | CPU |
| FAISS 检索 | 10万图库 | ~5ms | CPU |
| 批量导入 | 1000张 | ~60s | CPU |

---

## 常见问题

<details>
<summary><b>Q: 运行时提示找不到 libxcb-cursor0</b></summary>

**A:** 这是 Qt6 在 Linux 上的依赖：
```bash
sudo apt install libxcb-cursor0
```

</details>

<details>
<summary><b>Q: 中文搜索效果不好</b></summary>

**A:** 确保使用 CN-CLIP 模型而非原版 CLIP。运行 `scripts/export_cn_clip_onnx.py` 导出中文模型。

</details>

<details>
<summary><b>Q: 如何切换到 GPU 推理？</b></summary>

**A:**
1. 下载 ONNX Runtime GPU 版本
2. 安装 CUDA 和 cuDNN
3. 在代码中启用 CUDA Provider

</details>

<details>
<summary><b>Q: 模型文件在哪里下载？</b></summary>

**A:** 运行导出脚本自动下载：
```bash
cd scripts
python export_cn_clip_onnx.py --model ViT-B-16 --output ../assets/models
```

</details>

---

## 路线图

- [x] 以图搜图
- [x] 以文搜图（中英文）
- [x] 图文匹配
- [x] 图库管理
- [x] 中英文界面切换
- [x] 图像描述（BLIP）
- [ ] 视觉问答（VQA）
- [ ] GPU 加速
- [ ] 打包发布（AppImage / DMG / MSI）
- [ ] 插件系统

---

## 贡献

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

---

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 致谢

- [CN-CLIP](https://github.com/OFA-Sys/Chinese-CLIP) - 中文 CLIP 模型
- [Taiyi-BLIP](https://huggingface.co/IDEA-CCNL/Taiyi-BLIP-750M-Chinese) - 中文图像描述模型
- [FAISS](https://github.com/facebookresearch/faiss) - 向量相似度搜索
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) - 模型推理引擎
- [Qt](https://www.qt.io/) - 跨平台 GUI 框架
- [OpenCV](https://opencv.org/) - 计算机视觉库

---

<p align="center">
  Made with ❤️ by VIndex Team
</p>
