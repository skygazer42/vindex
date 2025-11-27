# 如何找到 ONNX Runtime 路径并配置 CMake

本项目需要找到 ONNX Runtime 的头文件和库路径，用于 `ONNXRUNTIME_ROOT`。下面给出常见系统的查找与验证方法。

## 1. 常见默认路径（先检查这些）
| 系统 | 可能的路径示例 |
| --- | --- |
| Linux | `/usr/local/onnxruntime`, `/usr/lib/onnxruntime`, `/opt/onnxruntime` |
| macOS | `/usr/local/onnxruntime`, `/opt/homebrew/onnxruntime` |
| Windows | `C:\libs\onnxruntime-win-x64-<version>`, `C:\Program Files\onnxruntime` |

如果你按 CMakeLists 中的默认配置，`ONNXRUNTIME_ROOT` 就是上述目录之一。

## 2. 直接搜索库文件定位
### Linux/macOS
```bash
# 查找 so/dylib
find /usr /opt -name "libonnxruntime*.so" -o -name "libonnxruntime*.dylib" 2>/dev/null
# 查找头文件
find /usr /opt -name "onnxruntime_cxx_api.h" 2>/dev/null
```

### Windows（PowerShell）
```powershell
Get-ChildItem -Recurse -Filter "onnxruntime*.dll" C:\ | Select-Object -First 5
Get-ChildItem -Recurse -Filter "onnxruntime_cxx_api.h" C:\ | Select-Object -First 5
```

找到库文件所在的目录（例如 `/usr/local/onnxruntime/lib/libonnxruntime.so`），则 `ONNXRUNTIME_ROOT` 为其上级目录（示例：`/usr/local/onnxruntime`）。

## 3. 验证目录结构
在确定的 `ONNXRUNTIME_ROOT` 下，至少应存在：
```
<ONNXRUNTIME_ROOT>/include/onnxruntime_cxx_api.h
<ONNXRUNTIME_ROOT>/lib/libonnxruntime.so   # Linux
<ONNXRUNTIME_ROOT>/lib/libonnxruntime.dylib # macOS
<ONNXRUNTIME_ROOT>/lib/onnxruntime.lib     # Windows
```

## 4. 配置 CMake
生成命令示例（替换为你找到的实际路径）：
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build build --config Release
```

Windows PowerShell：
```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release `
  -DONNXRUNTIME_ROOT=C:/libs/onnxruntime-win-x64-1.16.0
cmake --build build --config Release
```

## 5. 编译提示
- 如果编译提示 `Use of undeclared identifier 'OrtArenaAllocator'`，通常是因为编译器没找到 onnxruntime 的头文件；确认 `include` 路径正确。
- 若链接阶段找不到 `libonnxruntime`，确认库文件在 `<ONNXRUNTIME_ROOT>/lib` 并且 CMake 命令中的路径无误。

## 6. 依然找不到？
- 确认已安装 onnxruntime C/C++ 版（不是 Python 包）。
- 如通过包管理器安装（apt/brew），查看安装输出或包信息给出的安装前缀。
- 如果仍不确定，可以把 `find` 命令的输出发给我，我帮你定位。

