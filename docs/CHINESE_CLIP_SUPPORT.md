# 中文CLIP模型支持方案

**日期**: 2025-11-26
**版本**: v1.0
**状态**: 📋 设计阶段

---

## 📌 目标

为 VIndex 添加**中文CLIP模型支持**，实现：
- ✅ 中文文搜图（"一只猫"）
- ✅ 中英混合搜索
- ✅ 多模型切换
- ✅ 保持与现有英文CLIP兼容

---

## 🎯 推荐模型

### 模型 1: Taiyi-CLIP-Roberta-102M-Chinese

**来源**: [IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese](https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese)

**优势**:
- ✅ 专为中文优化
- ✅ 102M参数，模型较小
- ✅ 基于Roberta中文预训练
- ✅ 支持中文语义理解

**规格**:
- 文本编码器: Chinese-Roberta-wwm-ext-base-chinese
- 图像编码器: ViT-B/16
- 特征维度: 512
- 训练数据: 中文图文对

**适用场景**: 纯中文查询，中文图库

---

### 模型 2: CN-CLIP (ViT-B/16)

**来源**: [OFA-Sys/chinese-clip-vit-base-patch16](https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16)
**镜像**: [eisneim/cn-clip_vit-b-16](https://huggingface.co/eisneim/cn-clip_vit-b-16)

**优势**:
- ✅ 阿里达摩院出品
- ✅ 200M+中文图文对训练
- ✅ 中英双语支持
- ✅ 性能接近OpenAI CLIP

**规格**:
- 文本编码器: BERT-base-chinese
- 图像编码器: ViT-B/16
- 特征维度: 512
- 训练数据: Noah-Wukong + 自建数据集

**适用场景**: 中英混合查询，通用图库

---

### 模型对比

| 特性 | OpenAI CLIP | Taiyi-CLIP | CN-CLIP |
|------|-------------|------------|---------|
| **语言** | 英文 | 中文 | 中英双语 |
| **特征维度** | 768 (L/14) | 512 | 512 |
| **模型大小** | ~900MB | ~400MB | ~600MB |
| **中文性能** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **英文性能** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **推荐用途** | 英文/国际 | 纯中文 | 中英混合 |

**推荐选择**: **CN-CLIP** (中英双语，兼容性最好)

---

## 🏗️ 实现方案

### 方案 A: 多模型管理（推荐）

**架构设计**:
```
ModelManager (单例)
├── ClipEncoder (OpenAI CLIP - 英文)
├── ChineseClipEncoder (CN-CLIP - 中文)
└── TaiyiClipEncoder (Taiyi-CLIP - 中文)
```

**优势**:
- 支持多模型同时加载
- 用户可选择使用哪个模型
- 可以对比不同模型效果

**实现步骤**:
1. 创建 `ChineseClipEncoder` 类
2. 扩展 `ModelManager` 支持多编码器
3. 更新 GUI 添加模型选择下拉框
4. 创建模型下载/转换脚本

---

### 方案 B: 统一接口（简化）

**架构设计**:
```
ClipEncoder (基类)
├── OpenAIClipEncoder
├── CNClipEncoder
└── TaiyiClipEncoder
```

**优势**:
- 代码结构简单
- 易于维护
- 接口统一

**实现步骤**:
1. 重构 `ClipEncoder` 为抽象基类
2. 实现各个具体编码器
3. 运行时配置选择模型

---

## 📦 模型下载与转换

### 脚本 1: 下载中文CLIP模型

创建 `scripts/download_chinese_clip.py`:

```python
#!/usr/bin/env python3
"""
下载并转换中文CLIP模型到ONNX格式
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download
import torch
from transformers import BertTokenizer, BertModel, CLIPVisionModel
import onnx
from onnx import version_converter

def download_cn_clip(output_dir="./models/cn-clip"):
    """下载CN-CLIP模型"""
    print("📥 正在下载 CN-CLIP 模型...")

    # 下载模型文件
    model_path = snapshot_download(
        repo_id="OFA-Sys/chinese-clip-vit-base-patch16",
        cache_dir=output_dir,
        local_dir=output_dir,
        local_dir_use_symlinks=False
    )

    print(f"✅ 模型下载完成: {model_path}")
    return model_path

def download_taiyi_clip(output_dir="./models/taiyi-clip"):
    """下载Taiyi-CLIP模型"""
    print("📥 正在下载 Taiyi-CLIP 模型...")

    model_path = snapshot_download(
        repo_id="IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese",
        cache_dir=output_dir,
        local_dir=output_dir,
        local_dir_use_symlinks=False
    )

    print(f"✅ 模型下载完成: {model_path}")
    return model_path

def export_to_onnx(model_path, output_dir):
    """导出模型到ONNX格式"""
    print("🔄 正在转换为ONNX格式...")

    # TODO: 实现ONNX转换
    # 1. 加载PyTorch模型
    # 2. 导出视觉编码器
    # 3. 导出文本编码器
    # 4. 验证输出

    pass

def main():
    parser = argparse.ArgumentParser(description="下载中文CLIP模型")
    parser.add_argument("--model", choices=["cn-clip", "taiyi", "both"],
                       default="cn-clip", help="选择要下载的模型")
    parser.add_argument("--output", default="./assets/models",
                       help="输出目录")
    parser.add_argument("--export-onnx", action="store_true",
                       help="导出为ONNX格式")

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.model in ["cn-clip", "both"]:
        model_path = download_cn_clip(output_path / "cn-clip")
        if args.export_onnx:
            export_to_onnx(model_path, output_path / "cn-clip-onnx")

    if args.model in ["taiyi", "both"]:
        model_path = download_taiyi_clip(output_path / "taiyi-clip")
        if args.export_onnx:
            export_to_onnx(model_path, output_path / "taiyi-clip-onnx")

    print("🎉 全部完成！")

if __name__ == "__main__":
    main()
```

---

### 脚本 2: 导出CN-CLIP到ONNX

创建 `scripts/export_cn_clip_to_onnx.py`:

```python
#!/usr/bin/env python3
"""
将CN-CLIP模型导出为ONNX格式
"""

import torch
import onnx
from cn_clip.clip import load_from_name
import argparse
from pathlib import Path

def export_cn_clip_text_encoder(model, output_path):
    """导出文本编码器"""
    print("📤 导出文本编码器...")

    # 准备示例输入
    dummy_input = torch.randint(0, 21128, (1, 77))  # CN-CLIP vocab size
    dummy_attention_mask = torch.ones(1, 77)

    # 导出
    torch.onnx.export(
        model.text,
        (dummy_input, dummy_attention_mask),
        output_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['text_features'],
        dynamic_axes={
            'input_ids': {0: 'batch_size'},
            'attention_mask': {0: 'batch_size'},
            'text_features': {0: 'batch_size'}
        },
        opset_version=14,
        do_constant_folding=True
    )

    print(f"✅ 文本编码器已导出: {output_path}")

def export_cn_clip_visual_encoder(model, output_path):
    """导出视觉编码器"""
    print("📤 导出视觉编码器...")

    # 准备示例输入 (1, 3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224)

    # 导出
    torch.onnx.export(
        model.visual,
        dummy_input,
        output_path,
        input_names=['pixel_values'],
        output_names=['image_features'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size'},
            'image_features': {0: 'batch_size'}
        },
        opset_version=14,
        do_constant_folding=True
    )

    print(f"✅ 视觉编码器已导出: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ViT-B-16",
                       help="CN-CLIP模型名称")
    parser.add_argument("--output", default="./assets/models",
                       help="输出目录")
    parser.add_argument("--device", default="cpu",
                       help="设备 (cpu/cuda)")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print(f"📥 加载 CN-CLIP 模型: {args.model}")
    model, preprocess = load_from_name(args.model, device=args.device)
    model.eval()

    # 导出文本编码器
    text_output = output_dir / "cn_clip_text.onnx"
    export_cn_clip_text_encoder(model, text_output)

    # 导出视觉编码器
    visual_output = output_dir / "cn_clip_visual.onnx"
    export_cn_clip_visual_encoder(model, visual_output)

    print("🎉 导出完成！")
    print(f"  文本编码器: {text_output}")
    print(f"  视觉编码器: {visual_output}")

if __name__ == "__main__":
    main()
```

---

## 🔧 代码集成

### 1. 创建中文CLIP编码器类

`src/core/chinese_clip_encoder.h`:

```cpp
#pragma once

#include "clip_encoder.h"
#include <string>
#include <memory>

namespace vindex {
namespace core {

/**
 * @brief 中文CLIP编码器
 *
 * 支持中文文本编码和图像编码
 * 基于CN-CLIP或Taiyi-CLIP模型
 */
class ChineseClipEncoder : public ClipEncoder {
public:
    /**
     * @brief 构造函数
     * @param visualModelPath 视觉模型路径
     * @param textModelPath 文本模型路径
     * @param vocabPath 词表路径（BERT tokenizer）
     * @param embeddingDim 特征维度（默认512）
     */
    explicit ChineseClipEncoder(
        const std::string& visualModelPath,
        const std::string& textModelPath,
        const std::string& vocabPath,
        int embeddingDim = 512
    );

    ~ChineseClipEncoder() = default;

    /**
     * @brief 编码中文文本
     * @param text 中文文本（UTF-8编码）
     * @return 特征向量
     */
    std::vector<float> encodeText(const std::string& text) override;

    /**
     * @brief 批量编码中文文本
     */
    std::vector<float> encodeTextBatch(
        const std::vector<std::string>& texts
    ) override;

    /**
     * @brief 获取模型类型
     */
    std::string getModelType() const override { return "CN-CLIP"; }

private:
    // 使用BERT tokenizer而非BPE
    std::unique_ptr<class BertTokenizer> tokenizer_;
    int maxLength_;  // BERT默认512，CLIP通常用77
};

} // namespace core
} // namespace vindex
```

---

### 2. 扩展ModelManager支持多模型

`src/core/model_manager.h` 添加：

```cpp
class ModelManager {
public:
    // 现有方法...

    /**
     * @brief 获取中文CLIP编码器
     */
    ChineseClipEncoder& chineseClipEncoder();
    bool hasChineseClipEncoder() const;

    /**
     * @brief 获取当前激活的CLIP编码器
     * @return 英文或中文CLIP（根据配置）
     */
    ClipEncoder& activeClipEncoder();

    /**
     * @brief 设置激活模型
     * @param type "openai", "cn-clip", "taiyi"
     */
    void setActiveModel(const std::string& type);

    std::string getActiveModelType() const { return activeModelType_; }

private:
    void initializeChineseClipEncoder();

    std::unique_ptr<ChineseClipEncoder> chineseClipEncoder_;
    std::string activeModelType_{"openai"};  // 默认OpenAI CLIP
};
```

---

### 3. GUI更新：添加模型选择

在 `main_window.cpp` 添加模型选择下拉框：

```cpp
void MainWindow::setupToolBar() {
    QToolBar* toolbar = addToolBar("Main Toolbar");

    // 现有工具栏项...

    toolbar->addSeparator();

    // 模型选择下拉框
    toolbar->addWidget(new QLabel("CLIP Model:", this));

    modelSelector_ = new QComboBox(this);
    modelSelector_->addItem("OpenAI CLIP (English)", "openai");
    modelSelector_->addItem("CN-CLIP (中英双语)", "cn-clip");
    modelSelector_->addItem("Taiyi-CLIP (中文)", "taiyi");

    connect(modelSelector_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::onModelChanged);

    toolbar->addWidget(modelSelector_);
}

void MainWindow::onModelChanged(int index) {
    QString modelType = modelSelector_->itemData(index).toString();

    try {
        modelManager_->setActiveModel(modelType.toStdString());

        statusLabel_->setText(
            QString("Switched to %1")
            .arg(modelSelector_->currentText())
        );

    } catch (const std::exception& e) {
        QMessageBox::warning(
            this,
            "Error",
            QString("Failed to switch model: %1").arg(e.what())
        );
    }
}
```

---

## 📝 配置文件

创建 `assets/config/models.json`:

```json
{
  "models": {
    "openai-clip": {
      "name": "OpenAI CLIP",
      "language": "en",
      "visual_model": "clip_visual.onnx",
      "text_model": "clip_text.onnx",
      "vocab": "vocab/bpe_simple_vocab_16e6.txt",
      "tokenizer": "bpe",
      "embedding_dim": 768,
      "enabled": true
    },
    "cn-clip": {
      "name": "CN-CLIP",
      "language": "zh-cn,en",
      "visual_model": "cn_clip_visual.onnx",
      "text_model": "cn_clip_text.onnx",
      "vocab": "vocab/bert-base-chinese-vocab.txt",
      "tokenizer": "bert",
      "embedding_dim": 512,
      "enabled": false
    },
    "taiyi-clip": {
      "name": "Taiyi-CLIP",
      "language": "zh-cn",
      "visual_model": "taiyi_clip_visual.onnx",
      "text_model": "taiyi_clip_text.onnx",
      "vocab": "vocab/roberta-chinese-vocab.txt",
      "tokenizer": "roberta",
      "embedding_dim": 512,
      "enabled": false
    }
  },
  "default_model": "openai-clip"
}
```

---

## 🧪 测试计划

### 测试用例

| 场景 | 查询 | 预期结果 |
|------|------|----------|
| 中文查询 | "一只猫" | 返回猫的图片 |
| 英文查询 | "a cat" | 返回猫的图片 |
| 中英混合 | "红色的car" | 返回红色汽车 |
| 长句子 | "夕阳下的海滩" | 返回相关场景 |
| 专有名词 | "故宫" | 返回故宫图片 |

### 性能基准

| 模型 | 编码时间 (CPU) | 搜索时间 (10K) | 内存占用 |
|------|----------------|----------------|----------|
| OpenAI CLIP | ~50ms | ~10ms | ~900MB |
| CN-CLIP | ~40ms | ~10ms | ~600MB |
| Taiyi-CLIP | ~35ms | ~10ms | ~400MB |

---

## 📚 依赖更新

更新 `scripts/requirements.txt`:

```txt
# 现有依赖
torch>=2.0.0
onnx>=1.14.0
onnxruntime>=1.15.0
open-clip-torch>=2.20.0

# 新增：中文CLIP支持
cn_clip  # CN-CLIP官方包
transformers>=4.30.0  # BERT tokenizer
huggingface_hub>=0.16.0  # 模型下载
sentencepiece>=0.1.99  # 可选：更好的中文分词
```

---

## 📋 实施计划

### 第一阶段：基础支持（1-2天）
- [ ] 创建中文CLIP下载脚本
- [ ] 实现ONNX转换脚本
- [ ] 测试模型导出

### 第二阶段：代码集成（2-3天）
- [ ] 创建 `ChineseClipEncoder` 类
- [ ] 实现BERT tokenizer集成
- [ ] 扩展 `ModelManager` 多模型支持
- [ ] 添加配置文件加载

### 第三阶段：GUI更新（1天）
- [ ] 添加模型选择下拉框
- [ ] 更新搜索界面提示
- [ ] 添加语言自动检测

### 第四阶段：测试与优化（1-2天）
- [ ] 功能测试
- [ ] 性能基准测试
- [ ] 文档更新
- [ ] 示例和教程

**总计**: 5-8天

---

## 💡 额外优化

### 1. 自动语言检测

```cpp
std::string detectLanguage(const std::string& text) {
    // 简单实现：检测中文字符
    int chineseCount = 0;
    for (unsigned char c : text) {
        if (c >= 0x80) chineseCount++;  // 非ASCII
    }

    float ratio = static_cast<float>(chineseCount) / text.length();
    return ratio > 0.3 ? "zh" : "en";
}

ClipEncoder& ModelManager::autoSelectEncoder(const std::string& text) {
    std::string lang = detectLanguage(text);

    if (lang == "zh" && hasChineseClipEncoder()) {
        return chineseClipEncoder();
    } else {
        return clipEncoder();
    }
}
```

### 2. 混合搜索

支持同时使用多个模型搜索并合并结果：

```cpp
std::vector<SearchResult> DatabaseManager::hybridSearch(
    const std::string& query,
    int topK
) {
    // 使用两个模型分别搜索
    auto results1 = searchWithEncoder(query, clipEncoder(), topK);
    auto results2 = searchWithEncoder(query, chineseClipEncoder(), topK);

    // 合并并重新排序
    return mergeResults(results1, results2, topK);
}
```

### 3. 查询翻译

对于跨语言搜索，可以集成翻译API：

```cpp
std::string translateQuery(const std::string& text,
                          const std::string& targetLang) {
    // 调用翻译API（如百度翻译、Google Translate）
    // 实现查询翻译
    return translatedText;
}
```

---

## 📖 用户文档更新

添加到 `docs/QUICKSTART.md`:

### 使用中文CLIP模型

1. **下载模型**:
   ```bash
   cd scripts
   python download_chinese_clip.py --model cn-clip --export-onnx
   ```

2. **启动VIndex**:
   - 在工具栏选择 "CN-CLIP (中英双语)"

3. **中文搜索**:
   - 输入中文查询："一只可爱的猫"
   - 支持中英混合："红色的car"

4. **性能对比**:
   - OpenAI CLIP: 适合英文查询
   - CN-CLIP: 适合中英双语
   - Taiyi-CLIP: 适合纯中文

---

## 🎯 总结

添加中文CLIP支持将使VIndex成为**真正的多语言视觉搜索引擎**！

**优势**:
- 🌏 支持全球最大的中文用户群体
- 🔀 中英双语无缝切换
- 🚀 性能优异（模型更小更快）
- 🎨 更好的中文语义理解

**下一步**:
1. 确认是否开始实施
2. 选择优先集成的模型（推荐CN-CLIP）
3. 创建模型下载和转换脚本
4. 逐步集成到代码库

---

**维护者**: VIndex开发团队
**最后更新**: 2025-11-26
