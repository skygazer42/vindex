# 文搜图功能快速开始

## ✅ 已完成的工作

### 1. 核心组件

| 组件 | 状态 | 描述 |
|------|------|------|
| **TextTokenizer** | ✅ 已实现 | BPE文本分词器（简化版） |
| **ClipEncoder.encodeText()** | ✅ 已实现 | CLIP文本编码器 |
| **DatabaseManager.searchByText()** | ✅ 已实现 | 文搜图后端接口 |
| **TextSearchWidget** | ✅ 已实现 | 文搜图Qt界面 |
| **测试程序** | ✅ 已创建 | test_text_encoding.cpp |

### 2. TextSearchWidget 功能

- ✅ 多行文本输入（QTextEdit）
- ✅ 搜索参数配置（Top-K, 阈值）
- ✅ 快速示例按钮
- ✅ 搜索历史记录
- ✅ 结果网格展示
- ✅ 进度指示和状态更新

---

## 📋 下一步操作

### 步骤 1：集成到主窗口

修改 `src/gui/main_window.cpp`，添加文搜图标签页：

```cpp
#include "text_search_widget.h"  // 添加头文件

// 在 initializeDatabase() 函数中
void MainWindow::initializeDatabase() {
    // ... 现有代码 ...

    // 创建图搜图标签页（已有）
    imageSearchTab_ = new ImageSearchWidget(dbManager_.get(), this);
    tabWidget_->addTab(imageSearchTab_, "Image Search");

    // 创建文搜图标签页（新增）
    textSearchTab_ = new TextSearchWidget(dbManager_.get(), this);
    tabWidget_->addTab(textSearchTab_, "Text Search");

    // ... 现有代码 ...
}
```

### 步骤 2：准备 BPE 词表

确保词表文件存在：

```bash
cd assets/vocab

# 下载 BPE 词表
wget https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz

# 解压
gunzip bpe_simple_vocab_16e6.txt.gz

# 验证文件
ls -lh bpe_simple_vocab_16e6.txt
# 应该显示约 1.2MB 的文件
```

### 步骤 3：导出/验证 CLIP 文本模型

```bash
cd scripts

# 确保已安装 Python 依赖
pip install -r requirements.txt

# 导出 CLIP 模型（包含文本编码器）
python export_clip_to_onnx.py --model ViT-L-14 --pretrained openai --output ../assets/models

# 验证文件存在
ls -lh ../assets/models/clip_text.onnx
# 应该显示约 500MB+ 的文件
```

### 步骤 4：编译项目

```bash
cd ../build

# 重新配置（如有新文件）
cmake ..

# 编译
cmake --build . --config Release

# 或使用 make
make -j$(nproc)
```

### 步骤 5：测试文本编码（可选）

运行测试程序验证文本编码功能：

```bash
./test_text_encoding

# 或带自定义查询
./test_text_encoding "a cat" "sunset over ocean"
```

**预期输出：**
```
========================================
CLIP 文本编码测试程序
========================================

步骤 1: 配置模型路径...
  ✓ 模型目录: ./assets/models
  ✓ 词表路径: ./assets/vocab/bpe_simple_vocab_16e6.txt

步骤 2: 加载 CLIP 编码器...
  ✓ CLIP 编码器加载成功
  ✓ 特征维度: 768

步骤 3: 测试文本编码...

测试 1: "a cat"
--------------------------------------------------
  特征维度: 768
  向量模长: 1.000000
  前10个值: [0.1234, -0.5678, ...]
  ✓ 向量已正确归一化

...
```

### 步骤 6：运行应用测试文搜图

```bash
# 运行 VIndex
./VIndex

# 在应用中：
# 1. 导入一些图片（File → Import Folder）
# 2. 切换到 "Text Search" 标签页
# 3. 输入查询文本（如 "a cat"）
# 4. 点击 Search
# 5. 查看结果
```

---

## 🧪 测试用例

### 基础测试

| 测试描述 | 输入文本 | 期望行为 |
|---------|---------|---------|
| 单词查询 | `cat` | 返回猫的图片 |
| 短语查询 | `red car` | 返回红色汽车图片 |
| 完整句子 | `a person walking on the beach` | 返回相关场景 |
| 空查询 | `` | 显示错误提示 |
| 特殊字符 | `dog's toy` | 正确处理撇号 |

### 参数测试

| 参数 | 测试值 | 期望结果 |
|------|--------|----------|
| Top K | 5 | 返回5个结果 |
| Top K | 50 | 返回最多50个结果 |
| Threshold | 0.0 | 返回所有匹配 |
| Threshold | 0.8 | 只返回高相似度结果 |

### 功能测试

- ✅ 点击示例查询
- ✅ 使用搜索历史
- ✅ 清空输入
- ✅ 清空历史
- ✅ 结果图片点击
- ✅ 结果图片双击打开

---

## ⚠️ 常见问题

### Q1: "Text encoder not initialized"

**原因：** CLIP 文本模型未加载

**解决：**
1. 确认文件存在：`assets/models/clip_text.onnx`
2. 确认词表存在：`assets/vocab/bpe_simple_vocab_16e6.txt`
3. 检查文件权限

### Q2: "Search failed: Invalid token"

**原因：** BPE 分词器问题

**解决：**
- 当前使用简化版 BPE 分词器
- 对大多数英文查询有效
- 如遇问题，尝试简单的查询词

**长期方案：**
- 实现完整 BPE 算法
- 或使用 Python tokenizer 辅助

### Q3: 搜索结果不准确

**可能原因：**
1. 查询描述不够具体
2. 相似度阈值设置过高
3. 图库中无相关图片

**改进建议：**
- 使用更具体的描述（如 "white cat on wooden table" 而非 "cat"）
- 降低阈值（如 0.2-0.3）
- 增加 Top-K 值（如 20-50）

### Q4: 中文查询无效

**说明：**
- OpenAI CLIP 主要训练于英文数据
- 对常见中文词汇有限支持
- 建议使用英文查询

**如需中文支持：**
- 使用 Chinese-CLIP 模型
- 替换 ONNX 模型和分词器

---

## 🎯 性能优化建议

### 编码性能

| 场景 | 当前性能 | 优化方向 |
|------|---------|---------|
| 单次文本编码 | ~50ms (CPU) | GPU 加速 → <10ms |
| 批量编码 (10) | ~300ms | 批处理优化 |
| FAISS 搜索 (10K) | <10ms | 已优化 |

### 优化措施

1. **GPU 加速**
   ```cpp
   // 在 clip_encoder.cpp 中启用 CUDA
   sessionOptions_.AppendExecutionProvider_CUDA(0);
   ```

2. **缓存常用查询**
   ```cpp
   std::unordered_map<std::string, std::vector<float>> queryCache_;
   ```

3. **异步搜索**
   - 使用 Qt 线程
   - 避免 UI 阻塞

---

## 📊 评估指标

### 准确性评估

准备测试集：100 张图片 + 对应描述

计算指标：
- **Recall@10**：前10个结果中包含正确图片的比例
- **MRR**：正确结果的平均排名倒数

示例脚本：
```python
# scripts/evaluate_text_search.py
import json

ground_truth = [
    {"image": "cat_001.jpg", "query": "a white cat"},
    # ...
]

results = search_batch(ground_truth)
recall_at_10 = calculate_recall(results, k=10)
mrr = calculate_mrr(results)

print(f"Recall@10: {recall_at_10:.2%}")
print(f"MRR: {mrr:.3f}")
```

---

## 📖 用户文档片段

### 使用文搜图功能

1. **打开应用**
   - 启动 VIndex
   - 切换到 "Text Search" 标签页

2. **输入查询**
   - 在文本框中描述你要找的图片
   - 例如："a cat sitting on a table"
   - 或点击快速示例按钮

3. **调整参数（可选）**
   - **Top K**：返回结果数量（默认 10）
   - **Threshold**：相似度阈值（0-1，默认 0.3）

4. **执行搜索**
   - 点击 "Search" 按钮
   - 等待搜索完成（通常 < 1 秒）

5. **查看结果**
   - 结果按相似度排序
   - 显示相似度分数（百分比）
   - 单击查看详情，双击打开原图

6. **使用历史记录**
   - 点击历史列表中的查询
   - 自动重新执行搜索

---

## 🚀 后续增强

### 短期（1-2周）
- [ ] 完善 BPE 分词器
- [ ] 添加查询建议（自动完成）
- [ ] 批量查询功能
- [ ] 导出搜索结果

### 中期（1个月）
- [ ] 中文支持（Chinese-CLIP）
- [ ] 查询优化建议
- [ ] 搜索结果重排
- [ ] 语义相似查询推荐

### 长期（3个月+）
- [ ] 多语言支持
- [ ] 查询扩展（同义词）
- [ ] 负样本查询（"不包含..."）
- [ ] 组合查询（图+文）

---

## 📝 代码示例

### 使用文搜图 API

```cpp
#include "index/database_manager.h"

// 初始化数据库
DatabaseManager dbManager("./data/vindex.db", "./data/index/faiss.index");
dbManager.initialize();
dbManager.setEncoder(&ModelManager::instance().clipEncoder());

// 执行文搜图
auto results = dbManager.searchByText(
    "a cat on a table",  // 查询文本
    10,                  // Top-K
    0.3f                 // 阈值
);

// 处理结果
for (const auto& result : results) {
    std::cout << "ID: " << result.record.id << std::endl;
    std::cout << "Path: " << result.record.filePath << std::endl;
    std::cout << "Score: " << result.score << std::endl;
}
```

---

**参考文档：**
- [文搜图实现详解](TEXT_SEARCH_IMPLEMENTATION.md)
- [CLIP 论文](https://arxiv.org/abs/2103.00020)
- [项目主文档](../README.md)
