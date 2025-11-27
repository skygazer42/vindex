#include "caption_model.h"
#include <filesystem>
#include <stdexcept>
#include <sstream>

namespace vindex {
namespace core {

CaptionModel::CaptionModel(Ort::Env& env, const std::string& modelPath, int numThreads) {
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    sessionOptions_.SetIntraOpNumThreads(numThreads);

    if (!modelPath.empty() && std::filesystem::exists(modelPath)) {
#ifdef _WIN32
        std::wstring wPath(modelPath.begin(), modelPath.end());
        session_ = std::make_unique<Ort::Session>(env, wPath.c_str(), sessionOptions_);
#else
        session_ = std::make_unique<Ort::Session>(env, modelPath.c_str(), sessionOptions_);
#endif

        // 预拉取输入输出名称
        Ort::AllocatorWithDefaultOptions allocator;
        for (size_t i = 0; i < session_->GetInputCount(); ++i) {
            auto name = session_->GetInputNameAllocated(i, allocator);
            inputNamesStorage_.push_back(name.get());
            inputNames_.push_back(inputNamesStorage_.back().c_str());
        }
        for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
            auto name = session_->GetOutputNameAllocated(i, allocator);
            outputNamesStorage_.push_back(name.get());
            outputNames_.push_back(outputNamesStorage_.back().c_str());
        }
    }
}

std::string CaptionModel::generate(const cv::Mat& image, int /*maxLength*/) {
    if (!session_) {
        throw std::runtime_error("Caption model not loaded. Place blip_caption.onnx in assets/models.");
    }

    // 预处理图像
    std::vector<float> inputData = preprocessor_.preprocess(image);
    std::vector<int64_t> inputShape = preprocessor_.getInputShape();

    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo,
        inputData.data(),
        inputData.size(),
        inputShape.data(),
        inputShape.size()
    );

    // 运行
    auto outputs = session_->Run(
        Ort::RunOptions{nullptr},
        inputNames_.data(),
        &inputTensor,
        1,
        outputNames_.data(),
        outputNames_.size()
    );

    // 解析输出：优先处理字符串张量；否则如果是int64 token则拼接数字串
    const Ort::Value& out = outputs[0];
    auto typeInfo = out.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType dtype = typeInfo.GetElementType();

    if (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
        size_t totalBytes = out.GetStringTensorDataLength();
        std::string buffer(totalBytes, '\0');
        std::vector<size_t> offsets(typeInfo.GetElementCount());
        out.GetStringTensorContent(buffer.data(), buffer.size(), offsets.data(), offsets.size());
        // 只取第一个字符串
        return buffer.substr(0, buffer.find('\0'));
    }

    if (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        const int64_t* data = out.GetTensorData<int64_t>();
        size_t count = typeInfo.GetElementCount();
        std::ostringstream oss;
        for (size_t i = 0; i < count; ++i) {
            if (i) oss << ' ';
            oss << data[i];
        }
        return oss.str();
    }

    // 其他类型不支持，返回提示
    return "Caption model output type not supported.";
}

} // namespace core
} // namespace vindex
