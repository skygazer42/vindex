# VIndex - Vision/Language Indexing Suite

端到端的图像/文本检索与理解应用，整合 CLIP、BLIP、FAISS、SQLite 与 Qt6，支持图搜图、文搜图、图文匹配、图生文、图文问答等扩展场景。

## 功能与模型映射

| 功能 | 模型 | ONNX 文件 | 输入 → 输出 |
|------|------|-----------|-------------|
| 图搜图 | CLIP ViT-L/14 | `assets/models/clip_visual.onnx` | Image → 768D |
| 文搜图 | CLIP Text Encoder | `assets/models/clip_text.onnx` | Text → 768D |
| 图文匹配 | CLIP 双编码器 | 同上 | (Image, Text) → Score |
| 图生文 | BLIP2 / GIT | `assets/models/blip_caption.onnx` | Image → Text |
| 图文问答 | BLIP2-VQA | `assets/models/blip_vqa.onnx` | (Image, Question) → Answer |

## 仓库结构（扩展版）

```
vindex/
├── src/
│   ├── core/                 # 模型与预处理
│   │   ├── onnx_session.*    # ORT 会话管理
│   │   ├── clip_encoder.*    # CLIP 编码器
│   │   ├── caption_model.*   # 图生文
│   │   ├── vqa_model.*       # 图文问答
│   │   └── model_manager.*   # 模型单例管理
│   ├── index/                # 数据与索引
│   │   ├── faiss_index.*     # 向量检索封装
│   │   ├── id_mapping.*      # ID ↔ 路径映射
│   │   └── database_manager.*# SQLite 图库管理
│   ├── gui/                  # Qt6 界面
│   │   ├── main_window.*     # 主窗口 Tab
│   │   ├── image_search_widget.*  # 图搜图
│   │   ├── text_search_widget.*   # 文搜图
│   │   ├── match_widget.*         # 图文匹配
│   │   ├── caption_widget.*       # 图生文
│   │   ├── vqa_widget.*           # 问答
│   │   └── image_gallery.*        # 结果展示组件
│   ├── utils/                # 配置/日志/文件工具
│   └── main.cpp              # 应用入口
├── assets/
│   ├── models/               # ONNX 权重
│   ├── vocab/                # 词表
│   └── config/               # 应用配置
├── data/                     # 运行时数据（自动生成）
│   ├── image_database/
│   ├── index/ (faiss.index, id_map.db)
│   └── cache/
├── resources/                # Qt 资源（icons/styles/app.qrc）
├── scripts/                  # 模型导出与工具脚本
└── CMakeLists.txt
```

## 核心模块职责

- `core/model_manager.*` 单例管理 ORT 环境和所有模型实例，支持懒加载与预加载。
- `core/clip_encoder.*` 图像/文本编码、图文匹配，封装预处理与归一化。
- `index/faiss_index.*` 向量索引封装（新增/删除/批量/检索）。
- `index/database_manager.*` SQLite 元数据 + FAISS 同步；支持批量导入、重建索引。
- `gui/*_widget.*` 按功能划分的 Qt6 界面组件；`image_gallery.*` 复用结果网格。

## 开发阶段规划

- 阶段一：基础框架（CMake + ORT + OpenCV + FAISS + Qt 主窗口骨架）
- 阶段二：图搜图（ClipEncoder、FaissIndex、DatabaseManager、ImageSearchWidget 端到端）
- 阶段三：文搜图/图文匹配（CLIP 文本编码、BPE 分词器、TextSearchWidget）
- 阶段四：图生文 + VQA（BLIP 导出、CaptionModel/VQAModel、对应界面）
- 阶段五：完善与打包（图库管理、配置持久化、日志、错误处理、windeployqt/静态链接）

## 依赖

- Qt6 Widgets/Core/Gui/Sql
- OpenCV (core, imgproc, imgcodecs, highgui)
- ONNX Runtime (CPU 或 GPU)
- FAISS (CPU 版即可，GPU 可选)
- SQLite3
- 编译器：C++17 及以上

### 快速安装示例

#### Linux (Debian/Ubuntu)
```bash
sudo apt install qt6-base-dev libopencv-dev libsqlite3-dev
pip install onnxruntime  # 或下载官方 tar 包设置 ONNXRUNTIME_ROOT
conda install -c conda-forge faiss-cpu  # 或源码编译
```

#### Windows (vcpkg)
```powershell
.\vcpkg install qt6-base opencv4 sqlite3 faiss:x64-windows
# ONNX Runtime 手动下载解压，设置 ONNXRUNTIME_ROOT
```

## 模型与词表准备

1) CLIP 导出：`scripts/export_clip_to_onnx.py --model ViT-L-14 --pretrained openai`
2) 词表：下载 `bpe_simple_vocab_16e6.txt.gz` → 解压到 `assets/vocab/clip_vocab.txt`
3) BLIP/BLIP2/GIT/BLIP2-VQA：按各自转换脚本导出 ONNX，放入 `assets/models/`
4) 配置：`assets/config/app_config.json` 中可设置模型目录、索引路径、UI 选项等。

## 构建与运行

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build . --config Release
./VIndex   # Windows 下执行 VIndex.exe
```

常见 CMake 选项：
- `-DQt6_DIR`, `-DOpenCV_DIR`, `-DFAISS_DIR`, `-DSQLite3_DIR` 指向自定义安装。
- `-DUSE_CUDA=ON`（若在 CMakeLists 中开启）可切换 ORT/FAISS GPU。

## 数据与路径约定

- 模型：`assets/models/*.onnx`
- 词表：`assets/vocab/*.txt`
- 数据库：`data/index/id_map.db`
- 向量索引：`data/index/faiss.index`
- 缩略图缓存：`data/cache/`
- 图库根目录：`data/image_database/`

## 开发与测试建议

- 首先跑通图搜图：导出 CLIP、少量样本图、`Database → Import`、`Image Search` Tab。
- 批量导入时开启批处理（在 `database_manager.*` 中配置批大小）。
- 新模型接入：在 `model_manager.*` 注册，保持 ORT 环境共享以减少内存占用。
- 如需 GPU，加上 ORT CUDA provider 与 FAISS GPU 版本，注意 CUDA/cuDNN 兼容。

## 后续路线

- 增加增量索引持久化与崩溃恢复
- 增加中文/多语言 tokenizer 支持
- 引入检索重排（CLIP 互评或跨模态交互）
- 打包发布（windeployqt / macdeployqt / Linux AppImage）

## 数据流速览

- 图搜图：`QImage → cv::Mat 预处理 → CLIP Visual → 768D → FAISS 检索 → SQLite 取元数据 → UI 展示`
- 文搜图：`文本 → BPE Tokenizer → CLIP Text → 768D → FAISS 检索 → 元数据 → UI`
- 图文匹配：`图像 + 文本 → CLIP 双编码 → 相似度得分 → UI`
- 图生文 / VQA：`图像 (+ 问题) → BLIP/BLIP2 ONNX → 文本输出 → UI`

## 任务清单（执行顺序建议）

- [ ] 导出/校验 CLIP ONNX 与词表，补齐 `assets/models` 与 `assets/vocab`
- [ ] 打通 ORT + OpenCV + FAISS 编译链（CMake 可选 CUDA 开关）
- [ ] 实现 `ClipEncoder` 与 `FaissIndex`，写最小端到端图搜图 demo
- [ ] 接入 `DatabaseManager`，完成批量导入/删除/重建索引
- [ ] 完成 `ImageSearchWidget`，验证 UI 流程
- [ ] 接入文本检索与匹配（Tokenizer + TextSearchWidget）
- [ ] 接入 BLIP Caption/VQA，完善 UI Tab
- [ ] 增加配置持久化、日志、错误提示与加载进度
- [ ] 打包与发布脚本（windeployqt/macdeployqt/AppImage），补充用户文档

## 测试建议

- 单元：Tokenizer 分词一致性、向量归一化、FaissIndex 增删查、数据库 CRUD 与事务
- 集成：小样本图库（<100 图）端到端检索；大样本（>10k 图）构建与查询耗时
- UI：多平台（Win/Linux/macOS）窗口缩放、Tab 切换、导入/取消、加载时禁用按钮
- 性能：批处理导入、批量编码；GPU/CPU 结果一致性抽检；内存占用与索引尺寸线性检查

## 维护与贡献

- C++17，保持头/源一一对应；公共接口放在 `.h`，实现与私有函数放 `.cpp`
- 日志/错误通过 `utils/logger.*`，避免散落 `std::cout`
- 提交前运行：格式化（clang-format 若有配置）、最小功能自测；提交信息简洁、动词开头

## 扩展功能（TODO）

- [ ] 文搜图界面
- [ ] 图生文功能
- [ ] 图文问答（VQA）
- [ ] 分类管理
- [ ] 标签系统
- [ ] 批量操作
- [ ] 导出结果

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 📚 完整文档

- **[快速开始指南](docs/QUICKSTART.md)** - 5分钟上手教程
- **[详细编译指南](docs/BUILD.md)** - Windows/Linux/macOS 编译说明
- **[项目技术总结](docs/PROJECT_SUMMARY.md)** - 架构设计与实现细节

## 致谢

- OpenAI CLIP
- FAISS by Facebook Research
- ONNX Runtime by Microsoft
- Qt Framework
- OpenCV
