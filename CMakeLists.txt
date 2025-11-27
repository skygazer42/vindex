cmake_minimum_required(VERSION 3.20)
project(VIndex VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Qt配置
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# ============ 选项 ============
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

# ============ 查找依赖 ============

# Qt6
find_package(Qt6 REQUIRED COMPONENTS
    Core
    Gui
    Widgets
    Sql
)

# OpenCV
find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    imgcodecs
    highgui
)

# ONNX Runtime
# 注意：需要手动指定路径或通过环境变量
if(DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
elseif(WIN32)
    set(ONNXRUNTIME_ROOT "C:/libs/onnxruntime-win-x64-1.16.0")
else()
    set(ONNXRUNTIME_ROOT "/usr/local/onnxruntime")
endif()

if(EXISTS ${ONNXRUNTIME_ROOT})
    message(STATUS "Found ONNX Runtime at: ${ONNXRUNTIME_ROOT}")
    include_directories(${ONNXRUNTIME_ROOT}/include)
    link_directories(${ONNXRUNTIME_ROOT}/lib)
else()
    message(WARNING "ONNX Runtime not found at ${ONNXRUNTIME_ROOT}")
    message(WARNING "Please set ONNXRUNTIME_ROOT environment variable or install via package manager")
endif()

# FAISS
# 可以通过vcpkg或conda安装
find_package(faiss CONFIG)
if(NOT faiss_FOUND)
    message(STATUS "FAISS not found via CMake, trying manual configuration")

    if(DEFINED ENV{FAISS_ROOT})
        set(FAISS_ROOT $ENV{FAISS_ROOT})
        include_directories(${FAISS_ROOT}/include)
        link_directories(${FAISS_ROOT}/lib)
    else()
        message(WARNING "FAISS not found. Please install via vcpkg or conda, or set FAISS_ROOT")
    endif()
endif()

# SQLite3
find_package(SQLite3 REQUIRED)

# ============ 源文件 ============

# 核心模块
set(CORE_SOURCES
    src/core/image_preprocessor.cpp
    src/core/text_tokenizer.cpp
    src/core/clip_encoder.cpp
    src/core/model_manager.cpp
    src/core/onnx_session.cpp
    src/core/caption_model.cpp
    src/core/vqa_model.cpp
)

set(CORE_HEADERS
    src/core/image_preprocessor.h
    src/core/text_tokenizer.h
    src/core/clip_encoder.h
    src/core/model_manager.h
    src/core/onnx_session.h
    src/core/caption_model.h
    src/core/vqa_model.h
)

# 索引模块
set(INDEX_SOURCES
    src/index/faiss_index.cpp
    src/index/database_manager.cpp
    src/index/id_mapping.cpp
    src/index/text_corpus_index.cpp
)

set(INDEX_HEADERS
    src/index/faiss_index.h
    src/index/database_manager.h
    src/index/id_mapping.h
    src/index/text_corpus_index.h
)

# GUI模块
set(GUI_SOURCES
    src/gui/image_gallery.cpp
    src/gui/image_search_widget.cpp
    src/gui/text_search_widget.cpp
    src/gui/match_widget.cpp
    src/gui/image_to_text_widget.cpp
    src/gui/caption_widget.cpp
    src/gui/vqa_widget.cpp
    src/gui/database_widget.cpp
    src/gui/main_window.cpp
)

set(GUI_HEADERS
    src/gui/image_gallery.h
    src/gui/image_search_widget.h
    src/gui/text_search_widget.h
    src/gui/match_widget.h
    src/gui/image_to_text_widget.h
    src/gui/caption_widget.h
    src/gui/vqa_widget.h
    src/gui/database_widget.h
    src/gui/main_window.h
)

# 主程序
set(MAIN_SOURCES
    src/main.cpp
)

# ============ 可执行文件 ============
if(WIN32)
    add_executable(${PROJECT_NAME} WIN32
        ${MAIN_SOURCES}
        ${CORE_SOURCES}
        ${CORE_HEADERS}
        ${INDEX_SOURCES}
        ${INDEX_HEADERS}
        ${GUI_SOURCES}
        ${GUI_HEADERS}
    )
else()
    add_executable(${PROJECT_NAME}
        ${MAIN_SOURCES}
        ${CORE_SOURCES}
        ${CORE_HEADERS}
        ${INDEX_SOURCES}
        ${INDEX_HEADERS}
        ${GUI_SOURCES}
        ${GUI_HEADERS}
    )
endif()

# ============ 包含目录 ============
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${SQLite3_INCLUDE_DIRS}
)

# ============ 链接库 ============
target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::Sql
    ${OpenCV_LIBS}
    SQLite::SQLite3
    onnxruntime
)

# FAISS链接
if(faiss_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE faiss)
else()
    # 手动链接FAISS
    if(WIN32)
        target_link_libraries(${PROJECT_NAME} PRIVATE faiss)
    else()
        target_link_libraries(${PROJECT_NAME} PRIVATE faiss)
    endif()
endif()

# ============ 编译选项 ============
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4 /utf-8)
    # 解决MSVC的min/max宏冲突
    target_compile_definitions(${PROJECT_NAME} PRIVATE NOMINMAX)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic)
endif()

# ============ 安装配置 ============
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)

install(DIRECTORY assets/
    DESTINATION bin/assets
    FILES_MATCHING
    PATTERN "*.onnx"
    PATTERN "*.txt"
    PATTERN "*.json"
)

# Windows平台：部署Qt依赖
if(WIN32)
    # 获取Qt部署工具路径
    get_target_property(QMAKE_EXECUTABLE Qt6::qmake IMPORTED_LOCATION)
    get_filename_component(QT_BIN_DIR "${QMAKE_EXECUTABLE}" DIRECTORY)
    find_program(WINDEPLOYQT_EXECUTABLE windeployqt HINTS "${QT_BIN_DIR}")

    if(WINDEPLOYQT_EXECUTABLE)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND "${WINDEPLOYQT_EXECUTABLE}"
                --no-translations
                --no-system-d3d-compiler
                --no-opengl-sw
                "$<TARGET_FILE:${PROJECT_NAME}>"
            COMMENT "Running windeployqt..."
        )
    endif()
endif()

# ============ 测试程序 ============
add_executable(test_text_encoding
    src/test_text_encoding.cpp
    src/core/text_tokenizer.cpp
    src/core/clip_encoder.cpp
    src/core/image_preprocessor.cpp
    src/core/model_manager.cpp
    src/core/onnx_session.cpp
    src/core/caption_model.cpp
    src/core/vqa_model.cpp
)

target_include_directories(test_text_encoding PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(test_text_encoding PRIVATE
    ${OpenCV_LIBS}
    onnxruntime
)

if(faiss_FOUND)
    target_link_libraries(test_text_encoding PRIVATE faiss)
else()
    if(WIN32)
        target_link_libraries(test_text_encoding PRIVATE faiss)
    else()
        target_link_libraries(test_text_encoding PRIVATE faiss)
    endif()
endif()

# ============ 打印配置信息 ============
message(STATUS "")
message(STATUS "==================== Build Configuration ====================")
message(STATUS "Project: ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "")
message(STATUS "Dependencies:")
message(STATUS "  Qt6: ${Qt6_VERSION}")
message(STATUS "  OpenCV: ${OpenCV_VERSION}")
message(STATUS "  ONNX Runtime: ${ONNXRUNTIME_ROOT}")
message(STATUS "  SQLite3: ${SQLite3_VERSION}")
message(STATUS "===========================================================")
message(STATUS "")
