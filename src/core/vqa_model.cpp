#include "vqa_model.h"
#include <filesystem>
#include <stdexcept>
#include <sstream>
#include <numeric>

namespace vindex {
namespace core {

VqaModel::VqaModel(Ort::Env& env, const std::string& modelPath,
                   const std::string& vocabPath,
                   int contextLength,
                   int numThreads)
    : contextLength_(contextLength) {
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    sessionOptions_.SetIntraOpNumThreads(numThreads);

    if (!modelPath.empty() && std::filesystem::exists(modelPath)) {
#ifdef _WIN32
        std::wstring wPath(modelPath.begin(), modelPath.end());
        session_ = std::make_unique<Ort::Session>(env, wPath.c_str(), sessionOptions_);
#else
        session_ = std::make_unique<Ort::Session>(env, modelPath.c_str(), sessionOptions_);
#endif

        // 预取输入输出名称
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

    // 可选初始化 tokenizer
    if (!vocabPath.empty() && std::filesystem::exists(vocabPath)) {
        tokenizer_ = std::make_unique<TextTokenizer>(vocabPath, contextLength_);
    }
}

std::string VqaModel::answer(const cv::Mat& image, const std::string& question) {
    if (!session_) {
        throw std::runtime_error("VQA model not loaded. Place blip_vqa.onnx in assets/models.");
    }
    if (!tokenizer_) {
        throw std::runtime_error("Tokenizer not loaded. Provide blip/clip vocab for VQA.");
    }

    // 准备图像输入
    std::vector<float> imageData = preprocessor_.preprocess(image);
    std::vector<int64_t> imageShape = preprocessor_.getInputShape();

    // 文本 tokens
    std::vector<int64_t> tokens = tokenizer_->encode(question);
    std::vector<int64_t> textShape = {1, static_cast<int64_t>(tokenizer_->getContextLength())};

    // attention mask (1 for non-zero token)
    std::vector<int64_t> attention(tokens.size(), 0);
    for (size_t i = 0; i < tokens.size(); ++i) {
        attention[i] = tokens[i] == 0 ? 0 : 1;
    }

    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);
    auto imageTensor = Ort::Value::CreateTensor<float>(
        memInfo,
        imageData.data(),
        imageData.size(),
        imageShape.data(),
        imageShape.size()
    );
    auto textTensor = Ort::Value::CreateTensor<int64_t>(
        memInfo,
        tokens.data(),
        tokens.size(),
        textShape.data(),
        textShape.size()
    );
    auto attnTensor = Ort::Value::CreateTensor<int64_t>(
        memInfo,
        attention.data(),
        attention.size(),
        textShape.data(),
        textShape.size()
    );

    // 构建输入列表，尽量匹配常见命名
    std::vector<Ort::Value> inputTensors;
    std::vector<const char*> inputNames;
    for (const char* name : inputNames_) {
        std::string n(name);
        if (n.find("pixel") != std::string::npos || n.find("image") != std::string::npos) {
            inputNames.push_back(name);
            inputTensors.emplace_back(std::move(imageTensor));
        } else if (n.find("input_ids") != std::string::npos || n.find("text") != std::string::npos) {
            inputNames.push_back(name);
            inputTensors.emplace_back(std::move(textTensor));
        } else if (n.find("attention") != std::string::npos) {
            inputNames.push_back(name);
            inputTensors.emplace_back(std::move(attnTensor));
        }
    }
    // 如果自动匹配失败，回落到顺序绑定（按约定：image, input_ids, attention）
    if (inputTensors.empty()) {
        inputNames = inputNames_;
        inputTensors.clear();
        inputTensors.emplace_back(std::move(imageTensor));
        if (inputTensors.size() < inputNames.size()) inputTensors.emplace_back(std::move(textTensor));
        if (inputTensors.size() < inputNames.size()) inputTensors.emplace_back(std::move(attnTensor));
    }

    auto outputs = session_->Run(
        Ort::RunOptions{nullptr},
        inputNames.data(),
        inputTensors.data(),
        inputTensors.size(),
        outputNames_.data(),
        outputNames_.size()
    );

    const Ort::Value& out = outputs[0];
    auto typeInfo = out.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType dtype = typeInfo.GetElementType();

    if (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
        size_t totalBytes = out.GetStringTensorDataLength();
        std::string buffer(totalBytes, '\0');
        std::vector<size_t> offsets(typeInfo.GetElementCount());
        out.GetStringTensorContent(buffer.data(), buffer.size(), offsets.data(), offsets.size());
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

    return "VQA model output type not supported.";
}

} // namespace core
} // namespace vindex
