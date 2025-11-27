# Windows 完整构建指南

本文档详细介绍如何在 Windows 系统上从零开始构建 VIndex 项目。

## 目录

- [环境要求](#环境要求)
- [第一步：安装基础工具](#第一步安装基础工具)
- [第二步：安装 Qt6](#第二步安装-qt6)
- [第三步：安装 OpenCV](#第三步安装-opencv)
- [第四步：安装 ONNX Runtime](#第四步安装-onnx-runtime)
- [第五步：安装 SQLite3](#第五步安装-sqlite3)
- [第六步：安装 FAISS](#第六步安装-faiss)
- [第七步：配置环境变量](#第七步配置环境变量)
- [第八步：编译项目](#第八步编译项目)
- [第九步：导出 CLIP 模型](#第九步导出-clip-模型)
- [第十步：运行程序](#第十步运行程序)
- [常见问题](#常见问题)

---

## 环境要求

| 组件 | 最低版本 | 推荐版本 |
|------|----------|----------|
| Windows | 10 | 11 |
| Visual Studio | 2019 | 2022 |
| CMake | 3.20 | 3.28+ |
| Python | 3.8 | 3.10+ |

---

## 第一步：安装基础工具

### 1.1 安装 Visual Studio 2022

1. 下载 Visual Studio 2022 Community（免费）：
   - https://visualstudio.microsoft.com/zh-hans/downloads/

2. 安装时选择以下工作负载：
   - **"使用 C++ 的桌面开发"**

3. 在"单个组件"中确保勾选：
   - MSVC v143 - VS 2022 C++ x64/x86 生成工具
   - Windows 10/11 SDK
   - 用于 Windows 的 C++ CMake 工具

### 1.2 安装 CMake

1. 下载 CMake：
   - https://cmake.org/download/
   - 选择 `cmake-3.28.x-windows-x86_64.msi`

2. 安装时勾选 **"Add CMake to the system PATH"**

3. 验证安装：
   ```powershell
   cmake --version
   # 应显示: cmake version 3.28.x
   ```

### 1.3 安装 Git

1. 下载 Git for Windows：
   - https://git-scm.com/download/win

2. 安装时使用默认选项即可

3. 验证安装：
   ```powershell
   git --version
   ```

### 1.4 安装 Python

1. 下载 Python 3.10+：
   - https://www.python.org/downloads/

2. 安装时勾选：
   - **"Add Python to PATH"**
   - **"Install pip"**

3. 验证安装：
   ```powershell
   python --version
   pip --version
   ```

---

## 第二步：安装 Qt6

### 方法 A：使用 Qt 在线安装器（推荐）

1. 下载 Qt 在线安装器：
   - https://www.qt.io/download-qt-installer

2. 运行安装器，登录/注册 Qt 账号

3. 选择安装组件：
   - Qt 6.6.3（或最新 LTS）
     - ✅ MSVC 2019 64-bit
     - ✅ Qt 5 Compatibility Module
     - ✅ Additional Libraries > Qt Network Authorization

4. 安装路径建议：`C:\Qt`

5. 安装完成后，Qt6 位于：`C:\Qt\6.6.3\msvc2019_64`

### 方法 B：使用 aqtinstall（命令行）

```powershell
# 安装 aqtinstall
pip install aqtinstall

# 安装 Qt 6.6.3
aqt install-qt windows desktop 6.6.3 win64_msvc2019_64 -O C:\Qt

# Qt6 将安装到 C:\Qt\6.6.3\msvc2019_64
```

### 验证安装

```powershell
dir C:\Qt\6.6.3\msvc2019_64\bin\qmake.exe
# 应该存在该文件
```

---

## 第三步：安装 OpenCV

### 方法 A：使用预编译包（推荐）

1. 下载 OpenCV Windows 预编译包：
   - https://opencv.org/releases/
   - 选择 `opencv-4.10.0-windows.exe`（或最新版本）

2. 运行安装程序，解压到 `C:\libs\opencv`

3. 目录结构应为：
   ```
   C:\libs\opencv\
   ├── build\
   │   ├── x64\
   │   │   └── vc16\
   │   │       ├── bin\
   │   │       └── lib\
   │   ├── include\
   │   └── OpenCVConfig.cmake
   └── sources\
   ```

### 方法 B：使用 vcpkg

```powershell
# 安装 vcpkg（如果还没有）
git clone https://github.com/Microsoft/vcpkg.git C:\vcpkg
cd C:\vcpkg
.\bootstrap-vcpkg.bat

# 安装 OpenCV
.\vcpkg install opencv4:x64-windows
```

---

## 第四步：安装 ONNX Runtime

1. 下载 ONNX Runtime：
   - https://github.com/microsoft/onnxruntime/releases
   - 选择 `onnxruntime-win-x64-1.16.3.zip`（CPU 版本）
   - 如需 GPU 加速，选择 `onnxruntime-win-x64-gpu-1.16.3.zip`

2. 解压到 `C:\libs\onnxruntime`

3. 目录结构应为：
   ```
   C:\libs\onnxruntime\
   ├── include\
   │   └── onnxruntime\
   │       └── core\
   │           └── session\
   │               └── onnxruntime_cxx_api.h
   └── lib\
       ├── onnxruntime.dll
       └── onnxruntime.lib
   ```

4. 验证：
   ```powershell
   dir C:\libs\onnxruntime\include\onnxruntime_cxx_api.h
   dir C:\libs\onnxruntime\lib\onnxruntime.lib
   ```

---

## 第五步：安装 SQLite3

### 方法 A：使用 vcpkg（推荐）

```powershell
cd C:\vcpkg
.\vcpkg install sqlite3:x64-windows
```

### 方法 B：手动安装

1. 下载 SQLite 预编译 DLL：
   - https://www.sqlite.org/download.html
   - 下载 `sqlite-dll-win-x64-*.zip`
   - 下载 `sqlite-amalgamation-*.zip`（源码，用于头文件）

2. 创建目录并组织文件：
   ```
   C:\libs\sqlite3\
   ├── include\
   │   └── sqlite3.h      # 从 amalgamation 复制
   └── lib\
       ├── sqlite3.dll    # 从 dll 包复制
       └── sqlite3.lib    # 需要用 lib 工具生成
   ```

3. 生成 .lib 文件（在 VS 开发者命令提示符中）：
   ```powershell
   cd C:\libs\sqlite3\lib
   lib /def:sqlite3.def /out:sqlite3.lib /machine:x64
   ```

---

## 第六步：安装 FAISS

### 方法 A：使用 Conda（推荐）

1. 安装 Miniconda/Anaconda：
   - https://docs.conda.io/en/latest/miniconda.html

2. 创建环境并安装 FAISS：
   ```powershell
   conda create -n vindex python=3.10
   conda activate vindex
   conda install -c conda-forge faiss-cpu
   ```

3. 找到 FAISS 安装位置：
   ```powershell
   conda info --envs
   # 假设环境在 C:\Users\你的用户名\miniconda3\envs\vindex

   # FAISS 文件位于:
   # - 头文件: C:\Users\你的用户名\miniconda3\envs\vindex\Library\include\faiss
   # - 库文件: C:\Users\你的用户名\miniconda3\envs\vindex\Library\lib\faiss.lib
   ```

### 方法 B：使用 vcpkg

```powershell
cd C:\vcpkg
.\vcpkg install faiss:x64-windows
```

### 方法 C：从源码编译

```powershell
# 克隆 FAISS
git clone https://github.com/facebookresearch/faiss.git
cd faiss

# 创建构建目录
mkdir build
cd build

# 配置（仅 CPU）
cmake .. -G "Visual Studio 17 2022" -A x64 ^
    -DFAISS_ENABLE_GPU=OFF ^
    -DFAISS_ENABLE_PYTHON=OFF ^
    -DBUILD_TESTING=OFF ^
    -DCMAKE_INSTALL_PREFIX=C:\libs\faiss

# 编译
cmake --build . --config Release

# 安装
cmake --install . --config Release
```

---

## 第七步：配置环境变量

### 设置系统环境变量

打开 **系统属性 → 高级 → 环境变量**，添加以下变量：

| 变量名 | 值 |
|--------|-----|
| `Qt6_DIR` | `C:\Qt\6.6.3\msvc2019_64\lib\cmake\Qt6` |
| `OpenCV_DIR` | `C:\libs\opencv\build` |
| `ONNXRUNTIME_ROOT` | `C:\libs\onnxruntime` |
| `FAISS_ROOT` | `C:\libs\faiss` 或 Conda 环境路径 |
| `SQLite3_DIR` | `C:\vcpkg\installed\x64-windows\share\sqlite3` |

### 添加到 PATH

将以下路径添加到 `PATH` 环境变量：

```
C:\Qt\6.6.3\msvc2019_64\bin
C:\libs\opencv\build\x64\vc16\bin
C:\libs\onnxruntime\lib
C:\libs\faiss\bin（如果有）
```

### 使用 PowerShell 临时设置（每次构建前）

```powershell
$env:Qt6_DIR = "C:\Qt\6.6.3\msvc2019_64\lib\cmake\Qt6"
$env:OpenCV_DIR = "C:\libs\opencv\build"
$env:ONNXRUNTIME_ROOT = "C:\libs\onnxruntime"
$env:FAISS_ROOT = "C:\libs\faiss"
```

---

## 第八步：编译项目

### 8.1 克隆项目

```powershell
git clone <repository-url> vindex
cd vindex
```

### 8.2 创建构建目录

```powershell
mkdir build
cd build
```

### 8.3 配置 CMake

**方法 A：使用环境变量（推荐）**

```powershell
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release
```

**方法 B：手动指定路径**

```powershell
cmake .. -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DQt6_DIR=C:/Qt/6.6.3/msvc2019_64/lib/cmake/Qt6 ^
    -DOpenCV_DIR=C:/libs/opencv/build ^
    -DONNXRUNTIME_ROOT=C:/libs/onnxruntime ^
    -DFAISS_ROOT=C:/libs/faiss
```

**方法 C：使用 vcpkg 工具链**

```powershell
cmake .. -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake ^
    -DQt6_DIR=C:/Qt/6.6.3/msvc2019_64/lib/cmake/Qt6
```

### 8.4 编译

```powershell
cmake --build . --config Release --parallel
```

或者打开生成的 `VIndex.sln` 在 Visual Studio 中编译。

### 8.5 验证编译结果

```powershell
dir Release\VIndex.exe
# 应该看到生成的可执行文件
```

---

## 第九步：导出 CLIP 模型

### 9.1 安装 Python 依赖

```powershell
cd ..\scripts
pip install -r requirements.txt
```

如果没有 requirements.txt，手动安装：

```powershell
pip install torch torchvision
pip install cn-clip
pip install onnx onnxruntime
```

### 9.2 导出中文 CLIP 模型

```powershell
python export_cn_clip_onnx.py --model ViT-B-16 --output ../assets/models
```

这将生成：
- `assets/models/clip_visual.onnx` - 视觉编码器
- `assets/models/clip_text.onnx` - 文本编码器

### 9.3 下载词表

```powershell
# 创建词表目录
mkdir ..\assets\vocab

# 下载中文 CLIP 词表
# 从 https://github.com/OFA-Sys/Chinese-CLIP 获取
# 或使用项目自带的词表
```

---

## 第十步：运行程序

### 10.1 部署 Qt 依赖

```powershell
cd build\Release

# 使用 windeployqt 自动复制 Qt DLL
C:\Qt\6.6.3\msvc2019_64\bin\windeployqt.exe VIndex.exe
```

### 10.2 复制其他 DLL

确保以下 DLL 在 `Release` 目录或 `PATH` 中：

```powershell
# OpenCV DLL
copy C:\libs\opencv\build\x64\vc16\bin\opencv_world4100.dll .

# ONNX Runtime DLL
copy C:\libs\onnxruntime\lib\onnxruntime.dll .

# FAISS DLL（如果是动态链接）
copy C:\libs\faiss\bin\faiss.dll .
```

### 10.3 复制资源文件

```powershell
# 复制模型和词表
xcopy /E /I ..\..\assets .\assets
```

### 10.4 运行

```powershell
.\VIndex.exe
```

---

## 常见问题

### Q1: CMake 找不到 Qt6

**错误信息：**
```
Could not find a package configuration file provided by "Qt6"
```

**解决方案：**
```powershell
# 确保设置了 Qt6_DIR
$env:Qt6_DIR = "C:\Qt\6.6.3\msvc2019_64\lib\cmake\Qt6"

# 或在 cmake 命令中指定
cmake .. -DQt6_DIR=C:/Qt/6.6.3/msvc2019_64/lib/cmake/Qt6
```

### Q2: 找不到 ONNX Runtime

**错误信息：**
```
ONNX Runtime not found
```

**解决方案：**
```powershell
# 设置环境变量
$env:ONNXRUNTIME_ROOT = "C:\libs\onnxruntime"

# 确保目录结构正确
# C:\libs\onnxruntime\include\onnxruntime_cxx_api.h
# C:\libs\onnxruntime\lib\onnxruntime.lib
```

### Q3: FAISS 链接错误

**错误信息：**
```
LNK2019: unresolved external symbol faiss::...
```

**解决方案：**
1. 确保 FAISS 已正确安装
2. 检查是静态库还是动态库
3. 设置 `FAISS_ROOT` 环境变量
4. 如果使用 Conda，激活正确的环境

### Q4: 运行时找不到 DLL

**错误信息：**
```
无法找到 xxx.dll
```

**解决方案：**
```powershell
# 方法1：复制 DLL 到 exe 目录
copy C:\libs\onnxruntime\lib\*.dll .\Release\

# 方法2：添加到 PATH
$env:PATH += ";C:\libs\onnxruntime\lib;C:\libs\opencv\build\x64\vc16\bin"
```

### Q5: 应用程序无法正常启动 (0xc000007b)

**原因：** 32位/64位混用

**解决方案：**
1. 确保所有库都是 x64 版本
2. 使用 `dumpbin /headers xxx.dll` 检查 DLL 架构
3. 重新下载正确版本的依赖库

### Q6: 模型加载失败

**错误信息：**
```
Failed to load models: Model directory not found
```

**解决方案：**
1. 确保 `assets/models/` 目录存在
2. 运行模型导出脚本
3. 检查模型文件权限

### Q7: OpenCV 版本不匹配

**错误信息：**
```
The imported target "opencv_core" references the file ... but this file does not exist
```

**解决方案：**
```powershell
# 删除 CMake 缓存
rm -r build
mkdir build
cd build

# 重新配置，指定正确的 OpenCV 路径
cmake .. -DOpenCV_DIR=C:/libs/opencv/build
```

---

## 目录结构参考

成功安装后，依赖库的目录结构应该类似：

```
C:\
├── Qt\
│   └── 6.6.3\
│       └── msvc2019_64\
│           ├── bin\
│           ├── include\
│           └── lib\
│
├── libs\
│   ├── opencv\
│   │   └── build\
│   │       ├── include\
│   │       ├── x64\vc16\bin\
│   │       └── OpenCVConfig.cmake
│   │
│   ├── onnxruntime\
│   │   ├── include\
│   │   └── lib\
│   │
│   ├── faiss\
│   │   ├── include\
│   │   └── lib\
│   │
│   └── sqlite3\
│       ├── include\
│       └── lib\
│
└── vcpkg\  (可选)
    ├── installed\x64-windows\
    └── scripts\buildsystems\vcpkg.cmake
```

---

## 快速检查清单

在开始编译前，确认以下各项：

- [ ] Visual Studio 2022 已安装（含 C++ 桌面开发）
- [ ] CMake 3.20+ 已安装并在 PATH 中
- [ ] Git 已安装
- [ ] Python 3.8+ 已安装
- [ ] Qt6 已安装（MSVC 2019 64-bit）
- [ ] OpenCV 已安装
- [ ] ONNX Runtime 已下载解压
- [ ] SQLite3 已安装
- [ ] FAISS 已安装
- [ ] 环境变量已配置
- [ ] CLIP 模型已导出

---

## 联系与支持

如遇到问题：
1. 查看 [常见问题](#常见问题) 章节
2. 检查 CMake 输出的配置信息
3. 确保所有依赖版本兼容
4. 提交 Issue 并附上完整的错误日志
