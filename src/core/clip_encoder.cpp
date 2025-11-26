#include "clip_encoder.h"
#include <stdexcept>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace vindex {
namespace core {

ClipEncoder::ClipEncoder(const std::string& visualModelPath,
                         const std::string& textModelPath,
                         const std::string& vocabPath,
                         int embeddingDim)
    : embeddingDim_(embeddingDim)
    , memoryInfo_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    // 初始化图像预处理器
    imagePreprocessor_ = std::make_unique<ImagePreprocessor>();

    // 初始化ONNX会话
    initializeSessions(visualModelPath, textModelPath);

    // 如果提供了文本模型和词表，初始化文本分词器
    if (!textModelPath.empty() && !vocabPath.empty()) {
        // 默认 77（OpenAI CLIP）
        int contextLen = 77;

        // 如果模型输入固定长度，优先从 ONNX 形状读取
        if (textSession_) {
            auto typeInfo = textSession_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
            auto shape = typeInfo.GetShape();
            if (shape.size() >= 2 && shape[1] > 0) {
                contextLen = static_cast<int>(shape[1]);
            }
        }

        // 如果形状不明确，再按路径特征回退
        if (contextLen == 77) {
            std::string lowerPath = textModelPath;
            std::transform(lowerPath.begin(), lowerPath.end(), lowerPath.begin(), ::tolower);
            if (lowerPath.find("cn-clip-eisneim") != std::string::npos ||
                lowerPath.find("vit-b-16.txt") != std::string::npos) {
                contextLen = 52;
            }
        }

        textTokenizer_ = std::make_unique<TextTokenizer>(vocabPath, contextLen);
    }
}

void ClipEncoder::initializeSessions(const std::string& visualModelPath,
                                     const std::string& textModelPath) {
    // 创建ONNX Runtime环境
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ClipEncoder");

    // 配置会话选项
    sessionOptions_.SetIntraOpNumThreads(4);
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 加载视觉编码器
    if (!visualModelPath.empty()) {
#ifdef _WIN32
        std::wstring wPath(visualModelPath.begin(), visualModelPath.end());
        visualSession_ = std::make_unique<Ort::Session>(*env_, wPath.c_str(), sessionOptions_);
#else
        visualSession_ = std::make_unique<Ort::Session>(*env_, visualModelPath.c_str(), sessionOptions_);
#endif

        // 获取输入/输出名称
        Ort::AllocatorWithDefaultOptions allocator;

        size_t numInputNodes = visualSession_->GetInputCount();
        for (size_t i = 0; i < numInputNodes; i++) {
            auto inputName = visualSession_->GetInputNameAllocated(i, allocator);
            visualInputNamesStorage_.push_back(std::string(inputName.get()));
            visualInputNames_.push_back(visualInputNamesStorage_.back().c_str());
        }

        size_t numOutputNodes = visualSession_->GetOutputCount();
        for (size_t i = 0; i < numOutputNodes; i++) {
            auto outputName = visualSession_->GetOutputNameAllocated(i, allocator);
            visualOutputNamesStorage_.push_back(std::string(outputName.get()));
            visualOutputNames_.push_back(visualOutputNamesStorage_.back().c_str());
        }
    }

    // 加载文本编码器
    if (!textModelPath.empty()) {
#ifdef _WIN32
        std::wstring wPath(textModelPath.begin(), textModelPath.end());
        textSession_ = std::make_unique<Ort::Session>(*env_, wPath.c_str(), sessionOptions_);
#else
        textSession_ = std::make_unique<Ort::Session>(*env_, textModelPath.c_str(), sessionOptions_);
#endif

        // 获取输入/输出名称
        Ort::AllocatorWithDefaultOptions allocator;

        size_t numInputNodes = textSession_->GetInputCount();
        for (size_t i = 0; i < numInputNodes; i++) {
            auto inputName = textSession_->GetInputNameAllocated(i, allocator);
            textInputNamesStorage_.push_back(std::string(inputName.get()));
            textInputNames_.push_back(textInputNamesStorage_.back().c_str());
        }

        size_t numOutputNodes = textSession_->GetOutputCount();
        for (size_t i = 0; i < numOutputNodes; i++) {
            auto outputName = textSession_->GetOutputNameAllocated(i, allocator);
            textOutputNamesStorage_.push_back(std::string(outputName.get()));
            textOutputNames_.push_back(textOutputNamesStorage_.back().c_str());
        }
    }
}

// ==================== 图像编码 ====================

std::vector<float> ClipEncoder::encodeImage(const std::string& imagePath) {
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + imagePath);
    }
    return encodeImage(image);
}

std::vector<float> ClipEncoder::encodeImage(const cv::Mat& image) {
    // 预处理图像
    std::vector<float> imageData = imagePreprocessor_->preprocess(image);
    std::vector<int64_t> inputShape = imagePreprocessor_->getInputShape();

    // 运行推理
    return runVisualInference(imageData, inputShape);
}

std::vector<std::vector<float>> ClipEncoder::encodeImageBatch(const std::vector<cv::Mat>& images) {
    if (images.empty()) {
        return {};
    }

    // 预处理图像批次
    std::vector<float> imageData = imagePreprocessor_->preprocessBatch(images);
    std::vector<int64_t> inputShape = imagePreprocessor_->getBatchInputShape(images.size());

    // 运行推理
    std::vector<float> flatFeatures = runVisualInference(imageData, inputShape);

    // 重塑为批次格式
    std::vector<std::vector<float>> batchFeatures;
    batchFeatures.reserve(images.size());

    for (size_t i = 0; i < images.size(); ++i) {
        std::vector<float> features(flatFeatures.begin() + i * embeddingDim_,
                                   flatFeatures.begin() + (i + 1) * embeddingDim_);
        batchFeatures.push_back(std::move(features));
    }

    return batchFeatures;
}

std::vector<float> ClipEncoder::runVisualInference(const std::vector<float>& imageData,
                                                   const std::vector<int64_t>& inputShape) {
    if (!visualSession_) {
        throw std::runtime_error("Visual encoder not initialized");
    }

    // 创建输入tensor
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo_,
        const_cast<float*>(imageData.data()),
        imageData.size(),
        inputShape.data(),
        inputShape.size()
    );

    // 运行推理
    auto outputTensors = visualSession_->Run(
        Ort::RunOptions{nullptr},
        visualInputNames_.data(),
        &inputTensor,
        1,
        visualOutputNames_.data(),
        visualOutputNames_.size()
    );

    // 提取输出
    float* outputData = outputTensors[0].GetTensorMutableData<float>();
    auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

    size_t outputSize = 1;
    for (auto dim : outputShape) {
        outputSize *= dim;
    }

    std::vector<float> features(outputData, outputData + outputSize);

    // 对每个样本进行 L2 归一化（确保批次和单张处理结果一致）
    int batchSize = static_cast<int>(inputShape[0]);
    for (int i = 0; i < batchSize; ++i) {
        // 获取当前样本的特征向量范围
        auto start = features.begin() + i * embeddingDim_;
        auto end = start + embeddingDim_;

        // 计算 L2 范数
        float norm = std::sqrt(std::inner_product(start, end, start, 0.0f));

        // 归一化
        if (norm > 1e-8f) {
            for (auto it = start; it != end; ++it) {
                *it /= norm;
            }
        }
    }

    return features;
}

// ==================== 文本编码 ====================

std::vector<float> ClipEncoder::encodeText(const std::string& text) {
    if (!textSession_ || !textTokenizer_) {
        throw std::runtime_error("Text encoder not initialized");
    }

    // 分词
    std::vector<int64_t> tokens = textTokenizer_->encode(text);

    // 运行推理
    return runTextInference(tokens);
}

std::vector<std::vector<float>> ClipEncoder::encodeTextBatch(const std::vector<std::string>& texts) {
    if (!textSession_ || !textTokenizer_) {
        throw std::runtime_error("Text encoder not initialized");
    }

    if (texts.empty()) {
        return {};
    }

    // 批量分词
    std::vector<int64_t> allTokens = textTokenizer_->encodeBatch(texts);

    // 运行推理
    std::vector<float> flatFeatures = runTextInference(allTokens);

    // 重塑为批次格式
    std::vector<std::vector<float>> batchFeatures;
    batchFeatures.reserve(texts.size());

    for (size_t i = 0; i < texts.size(); ++i) {
        std::vector<float> features(flatFeatures.begin() + i * embeddingDim_,
                                   flatFeatures.begin() + (i + 1) * embeddingDim_);
        batchFeatures.push_back(std::move(features));
    }

    return batchFeatures;
}

std::vector<float> ClipEncoder::runTextInference(const std::vector<int64_t>& textTokens) {
    if (!textSession_) {
        throw std::runtime_error("Text encoder not initialized");
    }

    // 确定批次大小
    int batchSize = textTokens.size() / textTokenizer_->getContextLength();
    std::vector<int64_t> inputShape = {batchSize, textTokenizer_->getContextLength()};

    // 创建输入tensor
    Ort::Value inputTensor = Ort::Value::CreateTensor<int64_t>(
        memoryInfo_,
        const_cast<int64_t*>(textTokens.data()),
        textTokens.size(),
        inputShape.data(),
        inputShape.size()
    );

    // 注意：中文 CLIP 模型通常需要 attention_mask
    std::vector<Ort::Value> inputs;
    std::vector<const char*> inputNames;

    if (textInputNames_.size() >= 2) {
        // 构建 attention mask (非零为1)
        std::vector<int64_t> attention(textTokens.size(), 0);
        for (size_t i = 0; i < textTokens.size(); ++i) {
            attention[i] = textTokens[i] == 0 ? 0 : 1;
        }

        Ort::Value attnTensor = Ort::Value::CreateTensor<int64_t>(
            memoryInfo_,
            attention.data(),
            attention.size(),
            inputShape.data(),
            inputShape.size()
        );

        // 假设输入顺序：input_ids, attention_mask
        inputNames = {textInputNames_[0], textInputNames_[1]};
        inputs.emplace_back(std::move(inputTensor));
        inputs.emplace_back(std::move(attnTensor));
    } else {
        inputNames = textInputNames_;
        inputs.emplace_back(std::move(inputTensor));
    }

    // 运行推理
    auto outputTensors = textSession_->Run(
        Ort::RunOptions{nullptr},
        inputNames.data(),
        inputs.data(),
        inputs.size(),
        textOutputNames_.data(),
        textOutputNames_.size()
    );

    // 提取输出
    float* outputData = outputTensors[0].GetTensorMutableData<float>();
    auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

    size_t outputSize = 1;
    for (auto dim : outputShape) {
        outputSize *= dim;
    }

    std::vector<float> features(outputData, outputData + outputSize);

    // 对每个样本进行 L2 归一化（确保批次和单个文本处理结果一致）
    for (int i = 0; i < batchSize; ++i) {
        // 获取当前样本的特征向量范围
        auto start = features.begin() + i * embeddingDim_;
        auto end = start + embeddingDim_;

        // 计算 L2 范数
        float norm = std::sqrt(std::inner_product(start, end, start, 0.0f));

        // 归一化
        if (norm > 1e-8f) {
            for (auto it = start; it != end; ++it) {
                *it /= norm;
            }
        }
    }

    return features;
}

// ==================== 相似度计算 ====================

float ClipEncoder::computeSimilarity(const cv::Mat& image, const std::string& text) {
    auto imageFeatures = encodeImage(image);
    auto textFeatures = encodeText(text);
    return cosineSimilarity(imageFeatures, textFeatures);
}

float ClipEncoder::cosineSimilarity(const std::vector<float>& features1,
                                   const std::vector<float>& features2) {
    if (features1.size() != features2.size()) {
        throw std::invalid_argument("Feature vectors must have the same size");
    }

    // 计算点积
    float dotProduct = std::inner_product(features1.begin(), features1.end(),
                                         features2.begin(), 0.0f);

    // 如果特征已经归一化，点积就是余弦相似度
    // 否则需要除以向量模长
    float norm1 = std::sqrt(std::inner_product(features1.begin(), features1.end(),
                                              features1.begin(), 0.0f));
    float norm2 = std::sqrt(std::inner_product(features2.begin(), features2.end(),
                                              features2.begin(), 0.0f));

    if (norm1 < 1e-8 || norm2 < 1e-8) {
        return 0.0f;
    }

    // 余弦相似度范围[-1, 1]，转换到[0, 1]
    float similarity = dotProduct / (norm1 * norm2);
    return (similarity + 1.0f) / 2.0f;  // 归一化到[0, 1]
}

void ClipEncoder::normalizeL2(std::vector<float>& features) {
    float norm = std::sqrt(std::inner_product(features.begin(), features.end(),
                                             features.begin(), 0.0f));

    if (norm > 1e-8) {
        for (float& val : features) {
            val /= norm;
        }
    }
}

} // namespace core
} // namespace vindex
