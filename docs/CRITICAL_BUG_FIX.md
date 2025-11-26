# 🔴 严重Bug修复 - CN-CLIP集成

**日期**: 2025-11-26
**优先级**: **🚨 立即修复**

---

## 问题描述

### Bug 1: 上下文长度错误 🔴

**位置**: `src/core/clip_encoder.cpp` 第22-28行

**当前代码** (❌ 错误):
```cpp
// CN-CLIP 文本长度通常为512，标准 CLIP 为77
int contextLen = 77;
std::string lowerPath = textModelPath;
std::transform(lowerPath.begin(), lowerPath.end(), lowerPath.begin(), ::tolower);
if (lowerPath.find("cn-clip") != std::string::npos || lowerPath.find("vit-b-16") != std::string::npos) {
    contextLen = 512;  // ❌ 错误！应该是52
}
textTokenizer_ = std::make_unique<TextTokenizer>(vocabPath, contextLen);
```

**问题**:
- 代码设置 contextLen = 512
- 但eisneim CN-CLIP实际需要 **52 tokens**
- 这会导致维度不匹配，推理失败

**正确代码** (✅ 修复):
```cpp
// CN-CLIP (eisneim) 文本长度为52，标准 CLIP 为77
int contextLen = 77;
std::string lowerPath = textModelPath;
std::transform(lowerPath.begin(), lowerPath.end(), lowerPath.begin(), ::tolower);
if (lowerPath.find("cn-clip") != std::string::npos || lowerPath.find("eisneim") != std::string::npos) {
    contextLen = 52;  // ✅ eisneim CN-CLIP使用52个token
} else if (lowerPath.find("vit-b-16") != std::string::npos && lowerPath.find("cn") != std::string::npos) {
    contextLen = 52;  // ✅ 其他CN-CLIP变体也可能使用52
}
textTokenizer_ = std::make_unique<TextTokenizer>(vocabPath, contextLen);
```

---

### Bug 2: 注释错误

**位置**: `src/core/model_manager.cpp` 第14行

**当前代码** (❌ 错误):
```cpp
, embeddingDim_(512)  // 默认匹配中文 CN-CLIP (vit-b-16 输出512维)
```

**问题**:
- 注释正确（512维embedding是对的）
- 但不要与tokenizer的context_length混淆

**正确代码** (✅ 改进):
```cpp
, embeddingDim_(512)  // CN-CLIP特征维度512（注意：不是context length）
```

---

## 验证数据

### eisneim CN-CLIP实际输入要求

```python
# 图像编码器
输入: image [1, 3, 224, 224] (tensor(float))
输出: unnorm_image_features [1, 512] (tensor(float))

# 文本编码器 ⚠️ 注意这里
输入: text [1, 52] (tensor(int64))  # ← 52个token，不是77或512！
输出: unnorm_text_features [1, 512] (tensor(float))
```

### 不同模型对比

| 模型 | Context Length | Embedding Dim |
|------|----------------|---------------|
| **OpenAI CLIP** | 77 | 768 |
| **eisneim CN-CLIP** | **52** ⚠️ | 512 |
| **OFA CN-CLIP** | 77 (可能) | 512 |
| **Taiyi-CLIP** | 77 (可能) | 512 |

---

## 修复步骤

### 步骤1: 修复上下文长度

**文件**: `src/core/clip_encoder.cpp`

```cpp
// 第22-29行，修改为：
if (!textModelPath.empty() && !vocabPath.empty()) {
    // eisneim CN-CLIP使用52 tokens，标准CLIP使用77
    int contextLen = 77;
    std::string lowerPath = textModelPath;
    std::transform(lowerPath.begin(), lowerPath.end(), lowerPath.begin(), ::tolower);

    // 检测eisneim CN-CLIP模型
    if (lowerPath.find("eisneim") != std::string::npos ||
        lowerPath.find("vit-b-16.txt") != std::string::npos) {
        contextLen = 52;  // eisneim CN-CLIP特殊长度
    }

    textTokenizer_ = std::make_unique<TextTokenizer>(vocabPath, contextLen);
}
```

---

### 步骤2: 更新注释

**文件**: `src/core/model_manager.cpp`

```cpp
// 第14行，更新注释：
, embeddingDim_(512)  // CN-CLIP embedding维度512 (context length另外配置)
```

---

### 步骤3: 更新配置文件注释

**文件**: `assets/config/app_config.json`

```json
{
  "models": {
    "clip": {
      "visual_model": "assets/models/cn-clip-eisneim/vit-b-16.img.fp32.onnx",
      "text_model": "assets/models/cn-clip-eisneim/vit-b-16.txt.fp32.onnx",
      "vocab_path": "assets/vocab/clip_vocab.txt",
      "embedding_dim": 512,
      "context_length": 52,  // ← 添加这个说明
      "model_name": "CN-CLIP-ViT-B-16 (eisneim)",
      "note": "eisneim版本使用52 tokens，不是标准的77"
    }
  }
}
```

---

## 测试验证

### 测试1: 编译测试

```bash
cd /data/temp34/vindex/build
cmake ..
make -j$(nproc)

# 应该无编译错误
```

### 测试2: 简单推理测试

```cpp
// 创建测试文件 test_cn_clip.cpp
#include "core/model_manager.h"
#include <iostream>

int main() {
    auto& modelManager = vindex::core::ModelManager::instance();

    modelManager.setModelPath("./assets/models");
    modelManager.setVocabPath("./assets/vocab/clip_vocab.txt");
    modelManager.setEmbeddingDim(512);

    auto& encoder = modelManager.clipEncoder();

    // 测试文本编码
    try {
        std::cout << "测试中文文本编码..." << std::endl;
        auto features = encoder.encodeText("一只可爱的猫咪");
        std::cout << "✅ 成功！特征维度: " << features.size() << std::endl;

        if (features.size() != 512) {
            std::cerr << "❌ 错误：期望512维，实际" << features.size() << "维" << std::endl;
            return 1;
        }

        std::cout << "✅ 所有测试通过！" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "❌ 错误: " << e.what() << std::endl;
        return 1;
    }
}
```

### 测试3: 端到端文搜图测试

```bash
# 启动VIndex
./VIndex

# 在GUI中：
# 1. 切换到 Text Search 标签页
# 2. 输入查询："一只猫"
# 3. 点击 Search
# 4. 检查是否返回结果

# 期望：
# - 无错误提示
# - 返回相关图片
# - 相似度分数正常（0.0-1.0）
```

---

## 根本原因分析

### 为什么会出现这个bug？

1. **误解模型规格**
   - 可能看到512维embedding，误认为context length也是512
   - 实际上embedding维度和context length是两个不同的概念

2. **缺少模型文档检查**
   - 应该先用Python检查ONNX模型的实际输入形状
   - 再根据实际形状编写代码

3. **不同版本的差异**
   - eisneim版本针对中文优化，使用更短的context length (52)
   - 标准CLIP使用77
   - OpenAI CLIP-ViT-L使用77

### 正确的开发流程

1. **先验证模型** ✅
   ```python
   import onnxruntime as ort
   sess = ort.InferenceSession("model.onnx")
   for inp in sess.get_inputs():
       print(f"{inp.name}: {inp.shape}")
   ```

2. **再编写代码** ✅
   - 根据实际输入形状配置
   - 添加详细注释说明

3. **最后测试验证** ✅
   - 单元测试
   - 集成测试
   - 端到端测试

---

## 预防措施

### 1. 添加运行时检查

在 `ClipEncoder::initializeSessions()` 中添加：

```cpp
// 验证文本模型输入形状
if (textSession_) {
    Ort::AllocatorWithDefaultOptions allocator;
    auto inputTypeInfo = textSession_->GetInputTypeInfo(0);
    auto tensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    auto shape = tensorInfo.GetShape();

    if (shape.size() >= 2) {
        int64_t expectedLen = shape[1];
        int64_t actualLen = textTokenizer_->getContextLength();

        if (expectedLen > 0 && expectedLen != actualLen) {
            std::cerr << "⚠️  警告：Tokenizer长度(" << actualLen
                      << ")与模型期望(" << expectedLen << ")不匹配！" << std::endl;

            // 可选：自动调整或抛出异常
            throw std::runtime_error(
                "Context length mismatch: tokenizer=" + std::to_string(actualLen) +
                ", model=" + std::to_string(expectedLen)
            );
        }
    }
}
```

### 2. 添加文档注释

在关键位置添加清晰的注释：

```cpp
// ⚠️ 注意：不同CLIP模型的context length不同！
//   - OpenAI CLIP: 77 tokens
//   - eisneim CN-CLIP: 52 tokens  ← 特殊！
//   - 其他CN-CLIP可能: 77 tokens
// 请根据实际模型验证后配置
int contextLen = 77;  // 默认值
```

---

## 修复后的效果

### 修复前 (❌ 错误)
```
错误: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT :
Got invalid dimensions for input: text for the following indices
 index: 1 Got: 512 Expected: 52
```

### 修复后 (✅ 正常)
```
✅ 文本编码成功
✅ 特征维度: 512
✅ 相似度计算正常
✅ 文搜图功能可用
```

---

## 检查清单

修复前请确认：

- [ ] 已理解问题根源（512 vs 52）
- [ ] 已修改 clip_encoder.cpp 第27行
- [ ] 已更新相关注释
- [ ] 已添加运行时检查（可选）
- [ ] 重新编译项目
- [ ] 运行测试验证
- [ ] 更新相关文档

---

## 其他正确的配置

### ✅ 这些配置是正确的

1. **Embedding维度 = 512** ✅
   - CN-CLIP输出512维特征向量
   - 这个是正确的

2. **Attention mask生成** ✅
   - 代码中的attention mask生成逻辑正确
   - 非零token设为1，零token设为0

3. **双输入支持** ✅
   - 支持 input_ids + attention_mask
   - 逻辑正确

4. **L2归一化** ✅
   - 特征向量归一化
   - 实现正确

5. **模型路径** ✅
   - eisneim ONNX路径配置正确
   - 回退机制合理

### ❌ 只有这一个需要修复

- **Context length: 512 → 52** ❌
  - 这是唯一的关键bug
  - 必须立即修复

---

## 总结

**问题**: Context length设置为512，但模型需要52

**影响**:
- 🔴 **严重** - 文本编码完全无法工作
- 🔴 推理会立即失败
- 🔴 文搜图功能不可用

**修复**:
- 简单 - 只需改一个数字：512 → 52
- 快速 - 5分钟即可完成
- 关键 - 修复后功能立即可用

**验证**:
- 编译无误
- 推理成功
- 文搜图可用

---

**结论**: 代码整体架构和实现都很好，只有这**一个数字**需要修复！修复后即可正常使用CN-CLIP进行中文文搜图。

---

**维护者**: VIndex开发团队
**最后更新**: 2025-11-26
