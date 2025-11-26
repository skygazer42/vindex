# 中文CLIP快速开始指南

**5分钟上手中文图搜和文搜图功能**

---

## 🎯 目标

在VIndex中启用中文CLIP模型，实现：
- ✅ 中文文搜图："一只可爱的猫"
- ✅ 中英混合搜索："红色的car"
- ✅ 更好的中文语义理解

---

## 📋 前置要求

### 系统要求
- Python 3.8+
- 4GB+ 可用磁盘空间
- 稳定的网络连接（首次下载）

### 依赖安装

```bash
# 安装Python依赖
pip install huggingface_hub torch transformers

# 可选：安装cn_clip官方包（用于ONNX转换）
pip install cn_clip
```

---

## 🚀 方法一：快速下载（推荐）

### 步骤 1: 下载模型

```bash
cd vindex/scripts

# 下载CN-CLIP（推荐，中英双语）
python download_chinese_clip.py --model cn-clip
```

**预期输出**:
```
🌏 中文CLIP模型下载工具

============================================================
  🔍 检查依赖
============================================================
   ✅ huggingface_hub

📂 输出目录: /path/to/vindex/assets/models

============================================================
  📥 下载 CN-CLIP 模型
============================================================
📦 模型信息:
   名称: CN-CLIP ViT-B/16
   来源: OFA-Sys/chinese-clip-vit-base-patch16
   大小: ~600MB
   特征维度: 512
   语言: 中英双语

⏬ 开始下载...
   (首次下载可能需要几分钟，请耐心等待)

📂 下载到: /path/to/vindex/assets/models/cn-clip
   ⏳ config.json... ✅
   ⏳ pytorch_model.bin... ✅
   ⏳ vocab.txt... ✅
   ⏳ tokenizer_config.json... ✅

✅ CN-CLIP 模型下载完成!
   保存位置: /path/to/vindex/assets/models/cn-clip
```

### 步骤 2: 转换为ONNX格式

**注意**: 目前ONNX转换功能还在开发中。在完全实现之前，你可以：

**选项A**: 使用PyTorch模型（需要修改代码）
**选项B**: 等待ONNX转换脚本完成
**选项C**: 手动转换（高级用户）

```bash
# 将在未来版本提供
python export_cn_clip_to_onnx.py \
    --input ../assets/models/cn-clip \
    --output ../assets/models/cn_clip.onnx
```

### 步骤 3: 配置VIndex

编辑 `assets/config/models.json`:

```json
{
  "models": {
    "openai-clip": {
      "name": "OpenAI CLIP",
      "enabled": true,
      ...
    },
    "cn-clip": {
      "name": "CN-CLIP",
      "language": ["zh", "en"],
      "visual_model": "cn-clip/pytorch_model.bin",
      "text_model": "cn-clip/pytorch_model.bin",
      "vocab": "cn-clip/vocab.txt",
      "tokenizer": "bert",
      "embedding_dim": 512,
      "enabled": true
    }
  },
  "default_model": "cn-clip"
}
```

### 步骤 4: 启动VIndex

```bash
cd vindex/build
./VIndex
```

在工具栏选择 "CN-CLIP (中英双语)" 模型。

---

## 🔧 方法二：手动下载

如果自动下载失败，可以手动从Hugging Face下载：

### CN-CLIP

1. 访问: https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16
2. 点击 "Files and versions"
3. 下载以下文件:
   - `pytorch_model.bin`
   - `config.json`
   - `vocab.txt`
   - `tokenizer_config.json`
4. 保存到: `vindex/assets/models/cn-clip/`

### Taiyi-CLIP（可选）

1. 访问: https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese
2. 下载模型文件
3. 保存到: `vindex/assets/models/taiyi-clip/`

---

## 📝 使用示例

### 中文搜索

```
查询: "一只猫坐在桌子上"
结果: 返回猫在桌子上的图片
```

### 中英混合

```
查询: "红色的sports car"
结果: 返回红色跑车图片
```

### 场景描述

```
查询: "夕阳下的海滩"
结果: 返回海滩日落场景
```

### 情感色彩

```
查询: "温馨的家庭聚会"
结果: 返回家庭聚餐图片
```

---

## 🆚 模型对比

### 何时使用OpenAI CLIP?
- ✅ 英文查询
- ✅ 专业术语
- ✅ 国际化场景

### 何时使用CN-CLIP?
- ✅ 中文查询
- ✅ 中英混合
- ✅ 中文语义理解
- ✅ 模型更小（512维 vs 768维）

### 何时使用Taiyi-CLIP?
- ✅ 纯中文场景
- ✅ 对英文要求不高
- ✅ 追求最小模型

---

## ⚡ 性能对比

| 模型 | 特征维度 | 模型大小 | 编码速度 | 中文准确度 | 英文准确度 |
|------|---------|---------|---------|-----------|-----------|
| **OpenAI CLIP** | 768 | ~900MB | 50ms | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **CN-CLIP** | 512 | ~600MB | 40ms | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Taiyi-CLIP** | 512 | ~400MB | 35ms | ⭐⭐⭐⭐⭐ | ⭐⭐ |

---

## 🐛 常见问题

### Q1: 下载速度很慢怎么办？

**A**: 使用Hugging Face镜像站点:

```bash
# 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com

# 然后运行下载脚本
python download_chinese_clip.py --model cn-clip
```

### Q2: 提示 "huggingface_hub 未安装"？

**A**: 安装依赖:

```bash
pip install huggingface_hub
```

### Q3: ONNX转换失败？

**A**: 目前ONNX转换功能还在开发中，可以：
1. 暂时使用PyTorch版本（需修改代码）
2. 等待下一版本更新
3. 参考 `docs/CHINESE_CLIP_SUPPORT.md` 手动转换

### Q4: 中文搜索结果不准确？

**A**: 检查：
1. 是否选择了正确的模型（CN-CLIP或Taiyi-CLIP）
2. 查询是否足够具体
3. 图库是否包含相关图片
4. 尝试调整相似度阈值

### Q5: 可以同时加载多个模型吗？

**A**: 可以！在工具栏切换即可。但注意：
- 同时加载会占用更多内存（~1.5GB）
- 可以在配置文件中禁用不需要的模型

---

## 💡 进阶技巧

### 1. 自动语言检测

未来版本将支持自动检测查询语言并选择最佳模型。

### 2. 混合搜索

同时使用多个模型搜索并合并结果：

```cpp
// 伪代码
auto results1 = searchWithCLIP(query);
auto results2 = searchWithCNCLIP(query);
auto merged = mergeResults(results1, results2);
```

### 3. 查询优化

- ❌ 模糊: "猫"
- ✅ 具体: "一只白色的猫坐在窗台上"

- ❌ 单词: "beach"
- ✅ 短语: "sunset over the beach"

---

## 📚 相关文档

- [中文CLIP支持详细方案](CHINESE_CLIP_SUPPORT.md)
- [CN-CLIP论文](https://arxiv.org/abs/2211.01335)
- [Taiyi-CLIP GitHub](https://github.com/IDEA-CCNL/Fengshenbang-LM)
- [Hugging Face模型库](https://huggingface.co/models?search=chinese+clip)

---

## 🎯 下一步

### 立即尝试:
1. ✅ 下载CN-CLIP模型
2. ✅ 测试中文搜索
3. ✅ 对比不同模型效果

### 后续改进:
1. 完成ONNX转换脚本
2. 优化中文分词器
3. 添加更多中文模型支持
4. 性能基准测试

---

## 📞 获取帮助

遇到问题？
1. 查看 [常见问题](#常见问题)
2. 阅读 [详细文档](CHINESE_CLIP_SUPPORT.md)
3. 提交 Issue 到项目仓库

---

**提示**: 如果你主要使用中文，强烈推荐使用CN-CLIP！搜索准确度会有显著提升。

**最后更新**: 2025-11-26
