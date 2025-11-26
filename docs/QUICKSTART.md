# VIndex 快速开始指南

## 5分钟上手

### 1. 安装依赖（选择一种方式）

#### Ubuntu/Debian
```bash
sudo apt install qt6-base-dev libopencv-dev libsqlite3-dev cmake git python3-pip
conda install -c conda-forge faiss-cpu
```

#### Windows (使用 vcpkg)
```powershell
# 安装 vcpkg
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# 安装依赖
.\vcpkg install qt6:x64-windows opencv4:x64-windows sqlite3:x64-windows faiss:x64-windows
```

#### macOS
```bash
brew install cmake qt6 opencv sqlite faiss
```

### 2. 下载 ONNX Runtime

```bash
# Linux
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz
tar -xzf onnxruntime-linux-x64-1.16.0.tgz
export ONNXRUNTIME_ROOT=$PWD/onnxruntime-linux-x64-1.16.0

# Windows: 下载并解压到 C:\libs\onnxruntime-win-x64-1.16.0
# macOS: 下载 osx 版本并设置环境变量
```

### 3. 运行自动化脚本

#### Linux/macOS
```bash
cd vindex/scripts
chmod +x setup.sh
./setup.sh
```

#### Windows
```powershell
cd vindex\scripts
setup.bat
```

自动化脚本会：
- ✓ 安装 Python 依赖
- ✓ 导出 CLIP 模型为 ONNX 格式
- ✓ 下载 BPE 词表
- ✓ 编译项目

### 4. 运行应用

```bash
cd build
./VIndex  # Linux/macOS
# 或
.\Release\VIndex.exe  # Windows
```

### 5. 导入图像

1. 启动应用后，点击菜单 **File → Import Folder**
2. 选择包含图片的文件夹
3. 选择是否包含子文件夹
4. 等待导入完成

### 6. 开始搜索

1. 切换到 **Image Search** 标签页
2. 点击 **Select Image** 选择查询图片
3. 调整参数：
   - **Top K**: 返回结果数量（默认10）
   - **Threshold**: 相似度阈值（0.0-1.0）
4. 点击 **Search** 执行搜索
5. 查看结果：
   - 单击查看详情
   - 双击打开原图

---

## 故障排除

### 问题1：找不到 Qt6

```bash
# 设置 Qt6 路径
export Qt6_DIR=/path/to/qt6/lib/cmake/Qt6

# 或在 cmake 命令中指定
cmake .. -DQt6_DIR=/path/to/qt6/lib/cmake/Qt6
```

### 问题2：找不到 ONNX Runtime

```bash
# 设置环境变量
export ONNXRUNTIME_ROOT=/path/to/onnxruntime

# Windows
set ONNXRUNTIME_ROOT=C:\path\to\onnxruntime
```

### 问题3：模型加载失败

确认以下文件存在：
- `assets/models/clip_visual.onnx`
- `assets/models/clip_text.onnx`
- `assets/vocab/bpe_simple_vocab_16e6.txt`

如果不存在，运行：
```bash
cd scripts
python export_clip_to_onnx.py
```

### 问题4：搜索没有结果

1. 确认已导入图片（查看状态栏 "Images: N"）
2. 尝试降低阈值（设为 0.0）
3. 检查日志输出

---

## 高级选项

### 使用不同的 CLIP 模型

```bash
# 导出 ViT-B-32 (更快但精度稍低)
python export_clip_to_onnx.py --model ViT-B-32 --pretrained openai

# 修改 model_manager.cpp 中的 embeddingDim_
# ViT-B-32: 512
# ViT-L-14: 768
```

### 调整性能参数

编辑 `assets/config/app_config.json`:
```json
{
  "performance": {
    "num_threads": 8,      // 增加线程数
    "use_gpu": false,       // 启用 GPU（需要 GPU 版 ONNX Runtime）
    "cache_enabled": true
  },
  "database": {
    "batch_size": 64        // 增加批量大小
  }
}
```

### 重建索引

如果索引损坏或需要优化：
1. 菜单 → **Database → Rebuild Index**
2. 等待重建完成

---

## 性能建议

### 小型数据集（< 1000 张图）
- 默认配置即可
- 实时搜索 < 50ms

### 中型数据集（1K - 10K 张图）
- 增加 `num_threads` 到 8
- 使用批量导入
- 搜索时间 < 100ms

### 大型数据集（10K - 100K 张图）
- 考虑 GPU 加速
- 使用 FAISS IVF 索引（需修改代码）
- 定期重建索引优化
- 搜索时间 < 500ms

### 超大数据集（> 100K 张图）
- 必须使用 GPU
- 分布式部署
- 使用 FAISS GPU IVF 索引
- 考虑分片策略

---

## 下一步

探索更多功能：
- [ ] 文搜图（需实现文本搜索界面）
- [ ] 批量操作（删除、分类）
- [ ] 导出搜索结果
- [ ] 自定义模型

查看完整文档：
- `README.md` - 项目介绍
- `BUILD.md` - 详细编译指南
- `PROJECT_SUMMARY.md` - 技术细节

---

**遇到问题？** 查看 [常见问题](BUILD.md#常见问题) 或提交 Issue。
