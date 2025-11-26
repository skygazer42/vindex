#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include "image_preprocessor.h"
#include "text_tokenizer.h"

namespace vindex {
namespace core {

/**
 * @brief 图文问答模型占位
 */
class VqaModel {
public:
    VqaModel(Ort::Env& env, const std::string& modelPath,
             const std::string& vocabPath = "",
             int contextLength = 64,
             int numThreads = 4);
    ~VqaModel() = default;

    /**
     * @brief 回答图像问题
     */
    std::string answer(const cv::Mat& image, const std::string& question);

    bool loaded() const { return session_ != nullptr; }

private:
    std::unique_ptr<Ort::Session> session_;
    Ort::SessionOptions sessionOptions_;
    ImagePreprocessor preprocessor_;
    std::unique_ptr<TextTokenizer> tokenizer_;

    std::vector<std::string> inputNamesStorage_;
    std::vector<const char*> inputNames_;
    std::vector<std::string> outputNamesStorage_;
    std::vector<const char*> outputNames_;
    int contextLength_;
};

} // namespace core
} // namespace vindex
