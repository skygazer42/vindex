#!/bin/bash
# VIndex 自动化设置脚本（Linux/macOS）

set -e

echo "=================================="
echo "VIndex Setup Script"
echo "=================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检测操作系统
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
fi

echo -e "${GREEN}Detected OS: $OS${NC}"
echo ""

# 1. 检查依赖
echo "Step 1: Checking dependencies..."

check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 found"
        return 0
    else
        echo -e "${RED}✗${NC} $1 not found"
        return 1
    fi
}

MISSING_DEPS=0

check_command "cmake" || MISSING_DEPS=1
check_command "python3" || MISSING_DEPS=1
check_command "git" || MISSING_DEPS=1

if [ $MISSING_DEPS -eq 1 ]; then
    echo -e "${RED}Missing dependencies! Please install them first.${NC}"
    exit 1
fi

echo ""

# 2. 安装Python依赖
echo "Step 2: Installing Python dependencies..."
cd "$(dirname "$0")"
pip3 install -r requirements.txt
echo -e "${GREEN}✓ Python dependencies installed${NC}"
echo ""

# 3. 导出CLIP模型
echo "Step 3: Exporting CLIP models..."
if [ ! -f "../assets/models/clip_visual.onnx" ]; then
    echo "Exporting CLIP ViT-L/14..."
    python3 export_clip_to_onnx.py --model ViT-L-14 --pretrained openai --output ../assets/models
    echo -e "${GREEN}✓ Models exported${NC}"
else
    echo -e "${YELLOW}Models already exist, skipping...${NC}"
fi
echo ""

# 4. 下载词表
echo "Step 4: Downloading vocabulary..."
VOCAB_PATH="../assets/vocab/bpe_simple_vocab_16e6.txt"
if [ ! -f "$VOCAB_PATH" ]; then
    echo "Downloading BPE vocabulary..."
    wget -q --show-progress https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz -O /tmp/bpe_vocab.txt.gz
    gunzip /tmp/bpe_vocab.txt.gz
    mv /tmp/bpe_vocab.txt "$VOCAB_PATH"
    echo -e "${GREEN}✓ Vocabulary downloaded${NC}"
else
    echo -e "${YELLOW}Vocabulary already exists, skipping...${NC}"
fi
echo ""

# 5. 编译项目
echo "Step 5: Building project..."
cd ..
mkdir -p build
cd build

echo "Running CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

echo "Compiling..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo -e "${GREEN}✓ Build complete!${NC}"
echo ""

# 6. 完成
echo "=================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=================================="
echo ""
echo "To run VIndex:"
echo "  cd build"
echo "  ./VIndex"
echo ""
echo "To import images:"
echo "  1. Launch VIndex"
echo "  2. Go to File → Import Folder"
echo "  3. Select your image directory"
echo ""
