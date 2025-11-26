# 文搜图功能实现详解

## 📋 目录
- [功能概述](#功能概述)
- [技术原理](#技术原理)
- [当前实现状态](#当前实现状态)
- [待实现部分](#待实现部分)
- [实现步骤](#实现步骤)
- [测试计划](#测试计划)

---

## 功能概述

**文搜图（Text-to-Image Search）** 允许用户使用自然语言描述来搜索图像库中的相关图片。

### 使用场景示例

| 输入文本 | 期望结果 |
|---------|---------|
| "a dog playing in the park" | 返回公园里玩耍的狗的图片 |
| "sunset over the ocean" | 返回海洋日落的图片 |
| "red sports car" | 返回红色跑车的图片 |
| "person wearing glasses" | 返回戴眼镜的人的图片 |

---

## 技术原理

### CLIP 跨模态检索

CLIP (Contrastive Language-Image Pre-training) 通过对比学习，将图像和文本映射到同一个特征空间：

```
文本输入 "a cat on a table"
    ↓
BPE Tokenizer (分词)
    ↓
Token IDs: [49406, 320, 2368, 525, 320, 2904, 49407, 0, 0, ...]
    ↓
CLIP Text Encoder (ONNX)
    ↓
文本特征向量: [768维 float32]
    ↓
L2 归一化
    ↓
FAISS 向量检索
    ↓
Top-K 相似图像 ID
    ↓
SQLite 查询元数据
    ↓
返回结果 (图片路径 + 相似度分数)
```

### 关键技术点

1. **BPE 分词**
   - Byte-Pair Encoding
   - 词表大小：49,408
   - 上下文长度：77 tokens
   - 特殊 token：`<|startoftext|>` (49406), `<|endoftext|>` (49407)

2. **文本编码**
   - 输入：token IDs (batch_size, 77)
   - 输出：特征向量 (batch_size, 768)
   - 归一化：L2 norm

3. **跨模态匹配**
   - 图像和文本在同一特征空间
   - 使用余弦相似度计算匹配分数
   - 分数范围：0-1（越高越相似）

---

## 当前实现状态

### ✅ 已完成部分

#### 1. 文本分词器 (`src/core/text_tokenizer.h/cpp`)

**已实现：**
```cpp
class TextTokenizer {
    std::vector<int64_t> encode(const std::string& text);
    std::vector<int64_t> encodeBatch(const std::vector<std::string>& texts);
    std::string decode(const std::vector<int64_t>& tokens);
    // ...
};
```

**特点：**
- 基础文本清理
- SOT/EOT token 处理
- 固定长度输出（77）

**限制：**
- ⚠️ BPE 合并规则简化实现
- ⚠️ 需要完整的 BPE 词表文件

#### 2. CLIP 文本编码器 (`src/core/clip_encoder.h/cpp`)

**已实现：**
```cpp
class ClipEncoder {
    std::vector<float> encodeText(const std::string& text);
    std::vector<std::vector<float>> encodeTextBatch(
        const std::vector<std::string>& texts);
    float computeSimilarity(const cv::Mat& image, const std::string& text);
    // ...
};
```

**特点：**
- ONNX Runtime 推理
- 批量处理支持
- L2 归一化
- 余弦相似度计算

#### 3. 数据库管理器 (`src/index/database_manager.h/cpp`)

**已实现：**
```cpp
class DatabaseManager {
    std::vector<SearchResultWithRecord> searchByText(
        const std::string& queryText,
        int topK = 10,
        float threshold = 0.0f);
    // ...
};
```

**特点：**
- 端到端文搜图接口
- FAISS 向量检索
- SQLite 元数据查询
- 结果排序和过滤

---

## 待实现部分

### ❌ 未完成部分

#### 1. 完整 BPE 分词器

**当前问题：**
- 简化的分词实现
- 缺少完整的 BPE 合并规则
- 可能导致 token 化不准确

**解决方案：**

**方案A：使用 Python tokenizer（推荐）**
```python
# 利用 OpenAI CLIP 官方 tokenizer
import clip
tokenizer = clip.simple_tokenizer.SimpleTokenizer()
tokens = tokenizer.encode("a cat on a table")
```

优点：
- ✅ 完全准确
- ✅ 与训练时一致
- ✅ 简单可靠

缺点：
- ❌ 需要 Python 环境
- ❌ C++/Python 互调用

**方案B：完整 C++ 实现**

需要实现：
1. UTF-8 字节级编码
2. BPE 合并算法
3. 特殊字符处理

参考：https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py

**方案C：预分词（快速原型）**
```bash
# 预先生成 token IDs
python preprocess_text.py "a cat on a table" > tokens.txt
```

#### 2. TextSearchWidget 界面

**需要实现：**
```cpp
class TextSearchWidget : public QWidget {
    // 文本输入框
    QTextEdit* queryTextEdit_;

    // 搜索按钮
    QPushButton* searchBtn_;

    // 参数配置
    QSpinBox* topKSpinBox_;
    QLineEdit* thresholdEdit_;

    // 结果展示
    ImageGallery* resultGallery_;

    // 历史记录
    QListWidget* historyList_;
};
```

**UI 设计草图：**
```
┌─────────────────────────────────────────┐
│ Text Search                              │
├─────────────────────────────────────────┤
│ Query Text:                              │
│ ┌─────────────────────────────────────┐ │
│ │ a cat sitting on a table            │ │
│ └─────────────────────────────────────┘ │
│                                          │
│ Top K: [10 ▼]  Threshold: [0.3 ]        │
│                                          │
│ [Search]  [Clear]                        │
├─────────────────────────────────────────┤
│ Search History:                          │
│ • a dog in the park (10 results)        │
│ • sunset over ocean (25 results)        │
├─────────────────────────────────────────┤
│ Results (15 images found)                │
│ ┌───┐ ┌───┐ ┌───┐ ┌───┐               │
│ │img│ │img│ │img│ │img│               │
│ │95%│ │92%│ │88%│ │85%│               │
│ └───┘ └───┘ └───┘ └───┘               │
└─────────────────────────────────────────┘
```

---

## 实现步骤

### 阶段 1：验证现有文本编码（当前）

**目标：** 确认文本编码器可以正常工作

**步骤：**

1. **下载 BPE 词表**
```bash
cd assets/vocab
wget https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz
gunzip bpe_simple_vocab_16e6.txt.gz
```

2. **编写测试代码**
```cpp
// test_text_encoding.cpp
#include "core/clip_encoder.h"
#include <iostream>

int main() {
    // 初始化编码器
    ClipEncoder encoder(
        "assets/models/clip_visual.onnx",
        "assets/models/clip_text.onnx",
        "assets/vocab/bpe_simple_vocab_16e6.txt",
        768
    );

    // 测试文本编码
    std::string text = "a cat on a table";
    auto features = encoder.encodeText(text);

    std::cout << "Text: " << text << std::endl;
    std::cout << "Feature vector size: " << features.size() << std::endl;
    std::cout << "First 5 values: ";
    for (int i = 0; i < 5; i++) {
        std::cout << features[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

3. **验证输出**
- 特征向量维度应为 768
- 值应在 [-1, 1] 范围内（归一化后）
- 向量模长应接近 1.0

**预期问题：**
- 如果分词不准确，特征可能不正确
- 需要与 Python CLIP 对比验证

### 阶段 2：改进 BPE 分词器

**方案 1：Python 辅助（快速）**

创建辅助脚本：
```python
# scripts/tokenize_text.py
import clip
import sys
import json

tokenizer = clip.simple_tokenizer.SimpleTokenizer()

def tokenize(text):
    tokens = tokenizer.encode(text)
    return tokens.tolist()

if __name__ == "__main__":
    text = sys.argv[1]
    tokens = tokenize(text)
    print(json.dumps(tokens))
```

C++ 调用：
```cpp
std::vector<int64_t> tokenizeWithPython(const std::string& text) {
    std::string cmd = "python scripts/tokenize_text.py \"" + text + "\"";
    std::string result = exec(cmd);
    // 解析 JSON 返回的 tokens
    // ...
}
```

**方案 2：完整 C++ 实现（标准）**

参考 OpenAI CLIP tokenizer，实现：
1. `bytes_to_unicode()` - 字节映射
2. `get_pairs()` - 获取字符对
3. `bpe()` - BPE 合并算法

### 阶段 3：实现 TextSearchWidget

**文件：** `src/gui/text_search_widget.h/cpp`

**关键功能：**
1. 文本输入和编辑
2. 搜索参数配置
3. 搜索历史记录
4. 结果展示（复用 ImageGallery）
5. 多行文本支持
6. 搜索建议（可选）

### 阶段 4：集成到主窗口

**修改：** `src/gui/main_window.cpp`

```cpp
void MainWindow::initializeDatabase() {
    // ... 现有代码 ...

    // 创建文搜图标签页
    textSearchTab_ = new TextSearchWidget(dbManager_.get(), this);
    tabWidget_->addTab(textSearchTab_, "Text Search");

    // ... 现有代码 ...
}
```

### 阶段 5：端到端测试

**测试用例：**

| 测试 | 输入文本 | 期望行为 |
|------|---------|---------|
| 基础查询 | "cat" | 返回包含猫的图片 |
| 多词查询 | "red car" | 返回红色汽车图片 |
| 长句查询 | "a person walking on the beach at sunset" | 返回相关场景 |
| 特殊字符 | "dog's toy" | 正确处理撇号 |
| 空查询 | "" | 显示错误或返回空 |
| 中文查询 | "猫" | 是否支持（取决于 CLIP 模型）|

---

## 测试计划

### 单元测试

#### 1. TextTokenizer 测试
```cpp
TEST(TextTokenizerTest, BasicTokenization) {
    TextTokenizer tokenizer("assets/vocab/bpe_simple_vocab_16e6.txt");

    auto tokens = tokenizer.encode("hello world");

    EXPECT_EQ(tokens.size(), 77);  // 固定长度
    EXPECT_EQ(tokens[0], 49406);   // SOT token
    EXPECT_EQ(tokens[tokens.size()-1], 49407);  // EOT token
}
```

#### 2. CLIP Text Encoder 测试
```cpp
TEST(ClipEncoderTest, TextEncoding) {
    ClipEncoder encoder(...);

    auto features = encoder.encodeText("a cat");

    EXPECT_EQ(features.size(), 768);

    // 检查归一化
    float norm = 0.0f;
    for (auto v : features) norm += v * v;
    EXPECT_NEAR(sqrt(norm), 1.0f, 0.01f);
}
```

### 集成测试

#### 端到端文搜图流程
```cpp
TEST(TextSearchIntegrationTest, EndToEnd) {
    // 1. 初始化数据库
    DatabaseManager db("test.db", "test.index");
    db.initialize();

    // 2. 导入测试图片
    db.addImage("test_images/cat.jpg");
    db.addImage("test_images/dog.jpg");

    // 3. 执行文搜图
    auto results = db.searchByText("cat", 10, 0.0);

    // 4. 验证结果
    EXPECT_GT(results.size(), 0);
    EXPECT_EQ(results[0].record.filePath, "test_images/cat.jpg");
    EXPECT_GT(results[0].score, 0.5);  // 合理的相似度
}
```

### 性能测试

| 场景 | 输入 | 目标性能 |
|------|------|----------|
| 单次查询 | 短文本 | < 100ms |
| 批量查询 | 10个文本 | < 500ms |
| 大库检索 | 10K 图库 | < 200ms |

### 准确性测试

**数据集：** 准备 100 张测试图片 + 对应描述

**评估指标：**
- Recall@10：前10个结果中包含正确图片的比例
- MRR (Mean Reciprocal Rank)：正确结果的平均排名倒数

---

## 常见问题

### Q1: 为什么需要 BPE 分词？

**A:** CLIP 模型在训练时使用 BPE 分词，必须使用相同的分词方式才能得到正确的结果。

### Q2: 可以用其他分词器吗？

**A:** 不行。必须使用 CLIP 训练时的分词器（SimpleTokenizer with BPE）。

### Q3: 中文查询能工作吗？

**A:** 取决于 CLIP 模型是否在中文数据上训练。OpenAI CLIP 主要是英文，但对常见中文词汇有一定支持。如需完整中文支持，可使用 Chinese-CLIP。

### Q4: 如何提高搜索准确性？

**A:**
1. 使用更精确的描述（如 "a white cat sitting on a red sofa" 而非 "cat"）
2. 调整相似度阈值
3. 增加 Top-K 值
4. 使用更大的 CLIP 模型（如 ViT-L/14 而非 ViT-B/32）

---

## 下一步

1. ✅ 验证现有文本编码是否工作
2. ⬜ 改进 BPE 分词器（选择方案）
3. ⬜ 实现 TextSearchWidget 界面
4. ⬜ 集成到主窗口
5. ⬜ 端到端测试
6. ⬜ 性能优化
7. ⬜ 编写用户文档

---

**参考资源：**
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [CLIP GitHub](https://github.com/openai/CLIP)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [BPE Algorithm](https://arxiv.org/abs/1508.07909)
