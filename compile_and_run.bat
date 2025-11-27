@echo off
echo Compiling ONNX test...
g++ simple_onnx_test.cpp -o onnx_test.exe ^
    -I"C:\onnxruntime/include" ^
    -L"C:\onnxruntime/lib" ^
    -lonnxruntime -std=c++17

if errorlevel 1 (
    echo Compilation failed!
    pause
    exit /b 1
)

echo.
echo Copying DLL...
copy "C:\onnxruntime\bin\onnxruntime.dll" . >nul 2>&1

echo.
echo Running test...
onnx_test.exe
pause
