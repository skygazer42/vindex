#include "onnx_session.h"
#include <filesystem>
#include <stdexcept>

namespace vindex {
namespace core {

OnnxSession::OnnxSession() {
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    sessionOptions_.SetIntraOpNumThreads(4);
}

void OnnxSession::load(Ort::Env& env, const std::string& modelPath) {
    if (!std::filesystem::exists(modelPath)) {
        throw std::runtime_error("Model file not found: " + modelPath);
    }

#ifdef _WIN32
    std::wstring wPath(modelPath.begin(), modelPath.end());
    session_ = std::make_unique<Ort::Session>(env, wPath.c_str(), sessionOptions_);
#else
    session_ = std::make_unique<Ort::Session>(env, modelPath.c_str(), sessionOptions_);
#endif
}

} // namespace core
} // namespace vindex
