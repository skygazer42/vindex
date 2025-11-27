#include "caption_model.h"
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace fs = std::filesystem;

namespace vindex {
namespace core {

CaptionModel::CaptionModel(Ort::Env& env, const std::string& modelDir, int numThreads)
    : env_(&env)
    , memoryInfo_(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault))
{
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    sessionOptions_.SetIntraOpNumThreads(numThreads);

    if (modelDir.empty() || !fs::exists(modelDir)) {
        std::cerr << "BLIP model directory not found: " << modelDir << std::endl;
        return;
    }

    fs::path modelPath(modelDir);

    // 加载配置
    fs::path configPath = modelPath / "blip_config.json";
    if (fs::exists(configPath)) {
        loadConfig(configPath.string());
    }

    // 加载视觉编码器
    fs::path visualPath = modelPath / "blip_visual_encoder.onnx";
    if (fs::exists(visualPath)) {
        try {
#ifdef _WIN32
            std::wstring wPath = visualPath.wstring();
            visualEncoder_ = std::make_unique<Ort::Session>(*env_, wPath.c_str(), sessionOptions_);
#else
            visualEncoder_ = std::make_unique<Ort::Session>(*env_, visualPath.c_str(), sessionOptions_);
#endif
            // 获取输入输出名称
            Ort::AllocatorWithDefaultOptions allocator;
            for (size_t i = 0; i < visualEncoder_->GetInputCount(); ++i) {
                auto name = visualEncoder_->GetInputNameAllocated(i, allocator);
                visualInputNames_.push_back(name.get());
            }
            for (size_t i = 0; i < visualEncoder_->GetOutputCount(); ++i) {
                auto name = visualEncoder_->GetOutputNameAllocated(i, allocator);
                visualOutputNames_.push_back(name.get());
            }
            visualEncoderLoaded_ = true;
            std::cout << "BLIP visual encoder loaded: " << visualPath << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to load visual encoder: " << e.what() << std::endl;
        }
    }

    // 加载文本解码器
    fs::path decoderPath = modelPath / "blip_text_decoder.onnx";
    if (fs::exists(decoderPath)) {
        try {
#ifdef _WIN32
            std::wstring wPath = decoderPath.wstring();
            textDecoder_ = std::make_unique<Ort::Session>(*env_, wPath.c_str(), sessionOptions_);
#else
            textDecoder_ = std::make_unique<Ort::Session>(*env_, decoderPath.c_str(), sessionOptions_);
#endif
            // 获取输入输出名称
            Ort::AllocatorWithDefaultOptions allocator;
            for (size_t i = 0; i < textDecoder_->GetInputCount(); ++i) {
                auto name = textDecoder_->GetInputNameAllocated(i, allocator);
                decoderInputNames_.push_back(name.get());
            }
            for (size_t i = 0; i < textDecoder_->GetOutputCount(); ++i) {
                auto name = textDecoder_->GetOutputNameAllocated(i, allocator);
                decoderOutputNames_.push_back(name.get());
            }
            textDecoderLoaded_ = true;
            std::cout << "BLIP text decoder loaded: " << decoderPath << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to load text decoder: " << e.what() << std::endl;
        }
    }

    // 加载词表
    fs::path vocabPath = modelPath / "tokenizer" / "vocab.txt";
    if (fs::exists(vocabPath)) {
        loadVocab(vocabPath.string());
    }
}

bool CaptionModel::loadConfig(const std::string& configPath) {
    std::ifstream file(configPath);
    if (!file.is_open()) {
        return false;
    }

    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());

    // 简单 JSON 解析
    auto getValue = [&content](const std::string& key) -> std::string {
        size_t pos = content.find("\"" + key + "\"");
        if (pos == std::string::npos) return "";
        pos = content.find(":", pos);
        if (pos == std::string::npos) return "";
        pos++;
        while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t')) pos++;
        size_t end = content.find_first_of(",}\n", pos);
        std::string value = content.substr(pos, end - pos);
        // 去除引号
        if (!value.empty() && value.front() == '"') {
            value = value.substr(1);
            size_t q = value.find('"');
            if (q != std::string::npos) value = value.substr(0, q);
        }
        return value;
    };

    auto getInt = [&getValue](const std::string& key, int defaultVal) -> int {
        std::string v = getValue(key);
        return v.empty() ? defaultVal : std::stoi(v);
    };

    config_.imageSize = getInt("image_size", 384);
    config_.maxLength = getInt("max_length", 64);
    config_.vocabSize = getInt("vocab_size", 21128);
    config_.hiddenSize = getInt("hidden_size", 768);
    config_.bosTokenId = getInt("bos_token_id", 101);
    config_.eosTokenId = getInt("eos_token_id", 102);
    config_.padTokenId = getInt("pad_token_id", 0);

    std::cout << "BLIP config loaded: image_size=" << config_.imageSize
              << ", vocab_size=" << config_.vocabSize << std::endl;
    return true;
}

bool CaptionModel::loadVocab(const std::string& vocabPath) {
    std::ifstream file(vocabPath);
    if (!file.is_open()) {
        std::cerr << "Failed to open vocab file: " << vocabPath << std::endl;
        return false;
    }

    id2token_.clear();
    token2id_.clear();

    std::string line;
    int64_t id = 0;
    while (std::getline(file, line)) {
        // 去除末尾空白
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ')) {
            line.pop_back();
        }
        id2token_.push_back(line);
        token2id_[line] = id;
        id++;
    }

    std::cout << "Vocab loaded: " << id2token_.size() << " tokens" << std::endl;
    return true;
}

std::vector<float> CaptionModel::preprocessImage(const cv::Mat& image) {
    cv::Mat processed;

    // 调整大小
    cv::resize(image, processed, cv::Size(config_.imageSize, config_.imageSize));

    // BGR -> RGB
    cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);

    // 转为浮点并归一化到 [0, 1]
    processed.convertTo(processed, CV_32F, 1.0 / 255.0);

    // 标准化
    std::vector<cv::Mat> channels(3);
    cv::split(processed, channels);

    for (int c = 0; c < 3; ++c) {
        channels[c] = (channels[c] - config_.imageMean[c]) / config_.imageStd[c];
    }

    // 转为 NCHW 格式
    std::vector<float> result(3 * config_.imageSize * config_.imageSize);
    int idx = 0;
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < config_.imageSize; ++h) {
            for (int w = 0; w < config_.imageSize; ++w) {
                result[idx++] = channels[c].at<float>(h, w);
            }
        }
    }

    return result;
}

std::vector<float> CaptionModel::encodeImage(const cv::Mat& image) {
    if (!visualEncoderLoaded_) {
        throw std::runtime_error("Visual encoder not loaded");
    }

    // 预处理图像
    std::vector<float> inputData = preprocessImage(image);
    std::vector<int64_t> inputShape = {1, 3, config_.imageSize, config_.imageSize};

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo_,
        inputData.data(),
        inputData.size(),
        inputShape.data(),
        inputShape.size()
    );

    // 准备输入输出名称
    std::vector<const char*> inputNames;
    for (const auto& name : visualInputNames_) {
        inputNames.push_back(name.c_str());
    }
    std::vector<const char*> outputNames;
    for (const auto& name : visualOutputNames_) {
        outputNames.push_back(name.c_str());
    }

    // 运行视觉编码器
    auto outputs = visualEncoder_->Run(
        Ort::RunOptions{nullptr},
        inputNames.data(),
        &inputTensor,
        1,
        outputNames.data(),
        outputNames.size()
    );

    // 获取输出
    const Ort::Value& output = outputs[0];
    auto typeInfo = output.GetTensorTypeAndShapeInfo();
    auto shape = typeInfo.GetShape();
    size_t totalSize = 1;
    for (auto dim : shape) {
        totalSize *= dim;
    }

    const float* data = output.GetTensorData<float>();
    return std::vector<float>(data, data + totalSize);
}

std::vector<int64_t> CaptionModel::greedyDecode(const std::vector<float>& imageEmbeds, int maxLength) {
    if (!textDecoderLoaded_) {
        throw std::runtime_error("Text decoder not loaded");
    }

    std::vector<int64_t> generatedTokens;
    generatedTokens.push_back(config_.bosTokenId);

    // 计算 encoder hidden states 的形状
    // 假设 imageEmbeds 是 [batch, seq_len, hidden_size]
    int64_t batchSize = 1;
    int64_t encoderSeqLen = imageEmbeds.size() / config_.hiddenSize;
    std::vector<int64_t> encoderShape = {batchSize, encoderSeqLen, static_cast<int64_t>(config_.hiddenSize)};

    for (int step = 0; step < maxLength; ++step) {
        // 准备 input_ids
        std::vector<int64_t> inputIds = generatedTokens;
        std::vector<int64_t> inputIdsShape = {batchSize, static_cast<int64_t>(inputIds.size())};

        Ort::Value inputIdsTensor = Ort::Value::CreateTensor<int64_t>(
            memoryInfo_,
            inputIds.data(),
            inputIds.size(),
            inputIdsShape.data(),
            inputIdsShape.size()
        );

        // 准备 encoder_hidden_states
        Ort::Value encoderTensor = Ort::Value::CreateTensor<float>(
            memoryInfo_,
            const_cast<float*>(imageEmbeds.data()),
            imageEmbeds.size(),
            encoderShape.data(),
            encoderShape.size()
        );

        // 准备输入输出名称
        std::vector<const char*> inputNames;
        for (const auto& name : decoderInputNames_) {
            inputNames.push_back(name.c_str());
        }
        std::vector<const char*> outputNames;
        for (const auto& name : decoderOutputNames_) {
            outputNames.push_back(name.c_str());
        }

        // 运行解码器
        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(inputIdsTensor));
        inputs.push_back(std::move(encoderTensor));

        auto outputs = textDecoder_->Run(
            Ort::RunOptions{nullptr},
            inputNames.data(),
            inputs.data(),
            inputs.size(),
            outputNames.data(),
            outputNames.size()
        );

        // 获取 logits
        const Ort::Value& logits = outputs[0];
        auto logitsInfo = logits.GetTensorTypeAndShapeInfo();
        auto logitsShape = logitsInfo.GetShape();

        // logits 形状: [batch, seq_len, vocab_size]
        int64_t seqLen = logitsShape[1];
        int64_t vocabSize = logitsShape[2];

        const float* logitsData = logits.GetTensorData<float>();

        // 获取最后一个位置的 logits
        const float* lastLogits = logitsData + (seqLen - 1) * vocabSize;

        // Argmax
        int64_t nextToken = 0;
        float maxLogit = lastLogits[0];
        for (int64_t i = 1; i < vocabSize; ++i) {
            if (lastLogits[i] > maxLogit) {
                maxLogit = lastLogits[i];
                nextToken = i;
            }
        }

        // 检查是否结束
        if (nextToken == config_.eosTokenId) {
            break;
        }

        generatedTokens.push_back(nextToken);
    }

    return generatedTokens;
}

std::vector<int64_t> CaptionModel::beamSearchDecode(const std::vector<float>& imageEmbeds, int maxLength, int numBeams) {
    // 简化实现：当 numBeams=1 时退化为贪心解码
    if (numBeams <= 1) {
        return greedyDecode(imageEmbeds, maxLength);
    }

    // TODO: 实现完整的 beam search
    // 目前先使用贪心解码
    return greedyDecode(imageEmbeds, maxLength);
}

std::string CaptionModel::decodeTokens(const std::vector<int64_t>& tokens) {
    if (id2token_.empty()) {
        // 词表未加载，返回 token IDs
        std::ostringstream oss;
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i > 0) oss << " ";
            oss << tokens[i];
        }
        return oss.str();
    }

    std::string result;
    for (int64_t token : tokens) {
        // 跳过特殊 token
        if (token == config_.bosTokenId || token == config_.eosTokenId || token == config_.padTokenId) {
            continue;
        }

        if (token >= 0 && token < static_cast<int64_t>(id2token_.size())) {
            std::string tokenStr = id2token_[token];

            // 处理 BERT WordPiece 格式
            if (tokenStr.size() >= 2 && tokenStr[0] == '#' && tokenStr[1] == '#') {
                // 子词，直接拼接
                result += tokenStr.substr(2);
            } else if (tokenStr == "[UNK]") {
                result += "?";
            } else {
                // 新词，添加空格（中文不需要空格）
                // 简单判断：如果当前字符是 ASCII，且前一个也是 ASCII，则加空格
                if (!result.empty() && !tokenStr.empty()) {
                    bool prevIsAscii = (static_cast<unsigned char>(result.back()) < 128);
                    bool currIsAscii = (static_cast<unsigned char>(tokenStr[0]) < 128);
                    if (prevIsAscii && currIsAscii && result.back() != ' ') {
                        result += " ";
                    }
                }
                result += tokenStr;
            }
        }
    }

    return result;
}

std::string CaptionModel::generate(const cv::Mat& image, int maxLength, int numBeams) {
    if (!loaded()) {
        throw std::runtime_error("BLIP model not loaded. Please place blip_visual_encoder.onnx and blip_text_decoder.onnx in assets/models/blip/");
    }

    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    // 1. 编码图像
    std::vector<float> imageEmbeds = encodeImage(image);

    // 2. 解码生成文本
    std::vector<int64_t> tokens;
    if (numBeams > 1) {
        tokens = beamSearchDecode(imageEmbeds, maxLength, numBeams);
    } else {
        tokens = greedyDecode(imageEmbeds, maxLength);
    }

    // 3. 转换为文本
    return decodeTokens(tokens);
}

} // namespace core
} // namespace vindex
