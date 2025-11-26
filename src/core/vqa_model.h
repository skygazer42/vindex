#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>

namespace vindex {
namespace core {

/**
 * @brief 图文问答模型占位
 */
class VqaModel {
public:
    VqaModel(Ort::Env& env, const std::string& modelPath);
    ~VqaModel() = default;

    /**
     * @brief 回答图像问题
     */
    std::string answer(const cv::Mat& image, const std::string& question);

    bool loaded() const { return session_ != nullptr; }

private:
    std::unique_ptr<Ort::Session> session_;
    Ort::SessionOptions sessionOptions_;
};

} // namespace core
} // namespace vindex
