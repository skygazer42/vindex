# 编译指南

本文档详细说明了如何在不同平台上编译 VIndex。

## 目录

- [Windows编译](#windows编译)
- [Linux编译](#linux编译)
- [macOS编译](#macos编译)
- [常见问题](#常见问题)

---

## Windows编译

### 前置要求

- Visual Studio 2019 或更高版本
- CMake 3.20+
- Git

### 安装依赖

#### 方法1：使用vcpkg（推荐）

```powershell
# 1. 克隆vcpkg
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.bat

# 2. 安装依赖
./vcpkg install qt6:x64-windows
./vcpkg install opencv4:x64-windows
./vcpkg install sqlite3:x64-windows
./vcpkg install faiss:x64-windows

# 3. 集成vcpkg
./vcpkg integrate install
```

#### 方法2：手动安装

1. **Qt6**
   - 下载：https://www.qt.io/download
   - 安装Qt 6.5+
   - 设置环境变量：`set Qt6_DIR=C:\Qt\6.5.0\msvc2019_64\lib\cmake\Qt6`

2. **OpenCV**
   - 下载预编译版：https://opencv.org/releases/
   - 解压到 `C:\libs\opencv`
   - 设置环境变量：`set OpenCV_DIR=C:\libs\opencv\build`

3. **ONNX Runtime**
   - 下载：https://github.com/microsoft/onnxruntime/releases
   - 解压到 `C:\libs\onnxruntime-win-x64-1.16.0`
   - 设置环境变量：`set ONNXRUNTIME_ROOT=C:\libs\onnxruntime-win-x64-1.16.0`

4. **FAISS**
   - 使用vcpkg或conda：`conda install -c conda-forge faiss-cpu`

5. **SQLite3**
   - 通常随系统提供，或通过vcpkg安装

### 编译步骤

```powershell
# 1. 克隆项目
git clone <repository-url>
cd vindex

# 2. 创建构建目录
mkdir build
cd build

# 3. 配置（使用vcpkg）
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake ^
         -DCMAKE_BUILD_TYPE=Release

# 或手动指定路径
cmake .. -DCMAKE_BUILD_TYPE=Release ^
         -DQt6_DIR=C:/Qt/6.5.0/msvc2019_64/lib/cmake/Qt6 ^
         -DOpenCV_DIR=C:/libs/opencv/build ^
         -DONNXRUNTIME_ROOT=C:/libs/onnxruntime-win-x64-1.16.0

# 4. 编译
cmake --build . --config Release

# 5. 运行
cd Release
./VIndex.exe
```

---

## Linux编译

### Ubuntu/Debian

```bash
# 1. 安装依赖
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    qt6-base-dev \
    libopencv-dev \
    libsqlite3-dev

# 2. 安装FAISS
# 方法A：使用conda
conda install -c conda-forge faiss-cpu

# 方法B：从源码编译
git clone https://github.com/facebookresearch/faiss.git
cd faiss
cmake -B build . -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF
make -C build -j
sudo make -C build install

# 3. 安装ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz
tar -xzf onnxruntime-linux-x64-1.16.0.tgz
export ONNXRUNTIME_ROOT=$PWD/onnxruntime-linux-x64-1.16.0

# 4. 编译项目
git clone <repository-url>
cd vindex
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 5. 运行
./VIndex
```

### Arch Linux

```bash
# 1. 安装依赖
sudo pacman -S base-devel cmake git qt6-base opencv sqlite faiss

# 2. 安装ONNX Runtime
yay -S onnxruntime  # 或从AUR安装

# 3. 编译
git clone <repository-url>
cd vindex
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./VIndex
```

---

## macOS编译

### 使用Homebrew

```bash
# 1. 安装依赖
brew install cmake qt6 opencv sqlite faiss

# 2. 安装ONNX Runtime
# 下载macOS版本并解压
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-osx-x86_64-1.16.0.tgz
tar -xzf onnxruntime-osx-x86_64-1.16.0.tgz
export ONNXRUNTIME_ROOT=$PWD/onnxruntime-osx-x86_64-1.16.0

# 3. 编译
git clone <repository-url>
cd vindex
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DQt6_DIR=$(brew --prefix qt6)/lib/cmake/Qt6
make -j$(sysctl -n hw.ncpu)

# 4. 创建应用包
make install
./VIndex.app/Contents/MacOS/VIndex
```

---

## 导出CLIP模型

在运行应用前，需要导出CLIP模型：

```bash
cd scripts

# 安装Python依赖
pip install -r requirements.txt

# 导出模型
python export_clip_to_onnx.py --model ViT-L-14 --pretrained openai --output ../assets/models

# 下载词表
wget https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz
gunzip bpe_simple_vocab_16e6.txt.gz
mv bpe_simple_vocab_16e6.txt ../assets/vocab/
```

---

## 常见问题

### Q1: CMake找不到Qt6

**解决方案**：
```bash
# 设置Qt6路径
export Qt6_DIR=/path/to/qt6/lib/cmake/Qt6
# 或在cmake命令中指定
cmake .. -DQt6_DIR=/path/to/qt6/lib/cmake/Qt6
```

### Q2: 找不到ONNX Runtime

**解决方案**：
```bash
export ONNXRUNTIME_ROOT=/path/to/onnxruntime
```

### Q3: FAISS链接错误

**解决方案**：
- 确保FAISS正确安装
- 检查CMakeLists.txt中的FAISS路径
- 尝试使用vcpkg或conda安装

### Q4: 运行时找不到动态库

**Windows**：
```powershell
# 将DLL添加到PATH
set PATH=%PATH%;C:\libs\onnxruntime\lib;C:\Qt\6.5.0\msvc2019_64\bin
```

**Linux**：
```bash
# 添加到LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/onnxruntime/lib
```

### Q5: 编译时内存不足

**解决方案**：
```bash
# 限制并行编译任务数
make -j2  # 使用2个线程而不是全部
```

---

## 性能优化

### 启用优化编译

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native"
```

### GPU加速（可选）

1. 安装ONNX Runtime GPU版本
2. 安装CUDA版本的FAISS
3. 修改代码启用GPU

---

## 打包发布

### Windows

```powershell
# 使用windeployqt
cd build/Release
windeployqt VIndex.exe

# 创建安装包（使用NSIS或Inno Setup）
```

### Linux

```bash
# 创建AppImage
# 使用linuxdeployqt工具
```

### macOS

```bash
# 创建DMG
create-dmg VIndex.app
```
