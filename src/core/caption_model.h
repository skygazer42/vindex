#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include "image_preprocessor.h"

namespace vindex {
namespace core {

/**
 * @brief 图生文模型占位
 *
 * 当前仅做接口占位，后续可接入 BLIP/GIT ONNX。
 */
class CaptionModel {
public:
    CaptionModel(Ort::Env& env, const std::string& modelPath, int numThreads = 4);
    ~CaptionModel() = default;

    /**
     * @brief 生成描述
     */
    std::string generate(const cv::Mat& image, int maxLength = 64);

    bool loaded() const { return session_ != nullptr; }

private:
    std::unique_ptr<Ort::Session> session_;
    Ort::SessionOptions sessionOptions_;
    ImagePreprocessor preprocessor_;

    // 缓存输入输出名称，避免重复分配
    std::vector<std::string> inputNamesStorage_;
    std::vector<const char*> inputNames_;
    std::vector<std::string> outputNamesStorage_;
    std::vector<const char*> outputNames_;
};

} // namespace core
} // namespace vindex
