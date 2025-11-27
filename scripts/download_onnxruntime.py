#!/usr/bin/env python3
"""
Download and setup ONNX Runtime C++ SDK
"""
import os
import sys
import urllib.request
import zipfile
from pathlib import Path
import shutil

# Fix encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def download_onnxruntime():
    """Download ONNX Runtime C++ SDK"""
    # ONNX Runtime 1.16.0 for Windows x64
    url = "https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-win-x64-1.16.0.zip"

    # Local paths
    download_path = Path("onnxruntime.zip")
    extract_path = Path("onnxruntime-sdk")
    target_path = Path("C:/onnxruntime")

    print(f"📦 Downloading ONNX Runtime C++ SDK...")
    print(f"   URL: {url}")

    try:
        # Download if not exists
        if not download_path.exists():
            print(f"   Downloading... (this may take a few minutes)")
            urllib.request.urlretrieve(url, download_path)
            print(f"   ✅ Downloaded: {download_path.stat().st_size / (1024*1024):.1f} MB")
        else:
            print(f"   ✅ Using cached download")

        # Extract
        if not extract_path.exists():
            print(f"   Extracting...")
            with zipfile.ZipFile(download_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            # Rename extracted folder
            extracted = Path("onnxruntime-win-x64-1.16.0")
            if extracted.exists():
                extracted.rename(extract_path)
            print(f"   ✅ Extracted to {extract_path}")
        else:
            print(f"   ✅ Already extracted")

        # Setup in C:/onnxruntime (if admin) or local
        if not target_path.exists():
            try:
                # Try to create in C:/ (needs admin)
                target_path.mkdir(parents=True, exist_ok=True)
                # Copy contents
                for item in extract_path.iterdir():
                    dest = target_path / item.name
                    if item.is_dir():
                        shutil.copytree(item, dest, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, dest)
                print(f"   ✅ Installed to {target_path}")
                return str(target_path)
            except PermissionError:
                print(f"   ⚠️  Cannot install to C:/ (needs admin)")
                print(f"   Using local directory instead")
                return str(extract_path.absolute())
        else:
            print(f"   ✅ Already installed at {target_path}")
            return str(target_path)

    except Exception as e:
        print(f"   ❌ Error: {e}")
        # Fallback to local directory
        if extract_path.exists():
            return str(extract_path.absolute())
        return None

def main():
    print("=== ONNX Runtime C++ SDK Setup ===\n")

    onnx_path = download_onnxruntime()

    if onnx_path:
        print(f"\n✅ ONNX Runtime ready at: {onnx_path}")
        print(f"\n📝 Compile command:")
        print(f'g++ simple_onnx_test.cpp -o onnx_test.exe ^')
        print(f'    -I"{onnx_path}/include" ^')
        print(f'    -L"{onnx_path}/lib" ^')
        print(f'    -lonnxruntime -std=c++17')

        # Create batch file
        bat_content = f"""@echo off
echo Compiling ONNX test...
g++ simple_onnx_test.cpp -o onnx_test.exe ^
    -I"{onnx_path}/include" ^
    -L"{onnx_path}/lib" ^
    -lonnxruntime -std=c++17

if errorlevel 1 (
    echo Compilation failed!
    pause
    exit /b 1
)

echo.
echo Copying DLL...
copy "{onnx_path}\\bin\\onnxruntime.dll" . >nul 2>&1

echo.
echo Running test...
onnx_test.exe
pause
"""

        with open("compile_and_run.bat", "w") as f:
            f.write(bat_content)

        print(f"\n🚀 Quick compile & run:")
        print(f"   compile_and_run.bat")

        return onnx_path
    else:
        print(f"\n❌ Failed to setup ONNX Runtime")
        return None

if __name__ == "__main__":
    path = main()
    sys.exit(0 if path else 1)