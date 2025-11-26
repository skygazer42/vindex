#include "caption_model.h"
#include <filesystem>
#include <stdexcept>

namespace vindex {
namespace core {

CaptionModel::CaptionModel(Ort::Env& env, const std::string& modelPath) {
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    sessionOptions_.SetIntraOpNumThreads(4);

    if (!modelPath.empty() && std::filesystem::exists(modelPath)) {
#ifdef _WIN32
        std::wstring wPath(modelPath.begin(), modelPath.end());
        session_ = std::make_unique<Ort::Session>(env, wPath.c_str(), sessionOptions_);
#else
        session_ = std::make_unique<Ort::Session>(env, modelPath.c_str(), sessionOptions_);
#endif
    }
}

std::string CaptionModel::generate(const cv::Mat& /*image*/, int /*maxLength*/) {
    if (!session_) {
        throw std::runtime_error("Caption model not loaded. Place blip_caption.onnx in assets/models.");
    }

    // TODO: 实现 BLIP/GIT 推理逻辑。当前返回占位文本。
    return "Caption model loaded, decoding not yet implemented.";
}

} // namespace core
} // namespace vindex
