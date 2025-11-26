@echo off
REM VIndex 自动化设置脚本（Windows）

echo ==================================
echo VIndex Setup Script (Windows)
echo ==================================
echo.

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.8+
    pause
    exit /b 1
)
echo [OK] Python found

REM 检查CMake
cmake --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] CMake not found! Please install CMake 3.20+
    pause
    exit /b 1
)
echo [OK] CMake found

echo.
echo Step 1: Installing Python dependencies...
cd /d "%~dp0"
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install Python dependencies
    pause
    exit /b 1
)
echo [OK] Python dependencies installed
echo.

echo Step 2: Exporting CLIP models...
if not exist "..\assets\models\clip_visual.onnx" (
    echo Exporting CLIP ViT-L/14...
    python export_clip_to_onnx.py --model ViT-L-14 --pretrained openai --output ..\assets\models
    if errorlevel 1 (
        echo [ERROR] Failed to export models
        pause
        exit /b 1
    )
    echo [OK] Models exported
) else (
    echo [SKIP] Models already exist
)
echo.

echo Step 3: Downloading vocabulary...
if not exist "..\assets\vocab\bpe_simple_vocab_16e6.txt" (
    echo Downloading BPE vocabulary...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz' -OutFile '%TEMP%\bpe_vocab.txt.gz'"
    powershell -Command "Expand-Archive -Path '%TEMP%\bpe_vocab.txt.gz' -DestinationPath '..\assets\vocab\' -Force"
    if errorlevel 1 (
        echo [WARNING] Failed to download vocabulary automatically
        echo Please download manually from: https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz
        echo Extract to: ..\assets\vocab\bpe_simple_vocab_16e6.txt
    ) else (
        echo [OK] Vocabulary downloaded
    )
) else (
    echo [SKIP] Vocabulary already exists
)
echo.

echo Step 4: Building project...
cd ..
if not exist "build" mkdir build
cd build

echo Running CMake...
cmake .. -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 (
    echo [ERROR] CMake configuration failed
    echo.
    echo Please ensure all dependencies are installed:
    echo - Qt6 (set Qt6_DIR environment variable)
    echo - OpenCV (set OpenCV_DIR environment variable)
    echo - ONNX Runtime (set ONNXRUNTIME_ROOT environment variable)
    echo - FAISS (install via vcpkg or conda)
    echo - SQLite3 (usually included)
    echo.
    pause
    exit /b 1
)

echo Compiling...
cmake --build . --config Release
if errorlevel 1 (
    echo [ERROR] Build failed
    pause
    exit /b 1
)

echo.
echo ==================================
echo [SUCCESS] Setup Complete!
echo ==================================
echo.
echo To run VIndex:
echo   cd build\Release
echo   VIndex.exe
echo.
echo To import images:
echo   1. Launch VIndex
echo   2. Go to File -^> Import Folder
echo   3. Select your image directory
echo.

pause
