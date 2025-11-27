@echo off
echo ========================================
echo   VIndex C++ ONNX 集成编译脚本
echo ========================================
echo.

REM 设置路径
set ONNX_PATH=C:\onnxruntime

REM 检查ONNX Runtime
if not exist "%ONNX_PATH%" (
    echo [错误] ONNX Runtime 未找到在 %ONNX_PATH%
    echo 请运行: python download_onnxruntime.py
    pause
    exit /b 1
)

echo [1/4] 清理旧文件...
if exist onnx_test.exe del onnx_test.exe

echo [2/4] 编译 simple_test.cpp...
g++ simple_test.cpp -o onnx_test.exe -std=c++17
if errorlevel 1 (
    echo [错误] 编译失败!
    pause
    exit /b 1
)
echo       ✓ 编译成功

echo [3/4] 复制必要的DLL...
copy "%ONNX_PATH%\bin\*.dll" . >nul 2>&1

echo [4/4] 运行测试...
echo.
onnx_test.exe

if errorlevel 0 (
    echo.
    echo ========================================
    echo   ✓ C++ 测试成功！
    echo ========================================
) else (
    echo.
    echo [备用方案] 使用Python验证...
    python verify_cpp_integration.py
)

echo.
echo ========================================
echo   集成说明
echo ========================================
echo.
echo 在您的C++项目中使用ONNX模型:
echo.
echo 1. CMakeLists.txt 添加:
echo    set(ONNXRUNTIME_ROOT "C:/onnxruntime")
echo    find_package(onnxruntime REQUIRED)
echo    target_link_libraries(your_app onnxruntime)
echo.
echo 2. 代码中使用:
echo    #include ^<onnxruntime_cxx_api.h^>
echo    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "vindex");
echo    Ort::Session session(env, "model.onnx", options);
echo.
echo 3. 部署时包含:
echo    - onnxruntime.dll
echo    - 所有ONNX模型文件
echo.
pause