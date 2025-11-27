#include "vqa_model.h"
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

VqaModel::VqaModel(Ort::Env& env, const std::string& modelDir, int numThreads)
    : env_(&env)
    , memoryInfo_(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault))
{
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    sessionOptions_.SetIntraOpNumThreads(numThreads);

    if (modelDir.empty() || !fs::exists(modelDir)) {
        std::cerr << "BLIP VQA model directory not found: " << modelDir << std::endl;
        return;
    }

    fs::path modelPath(modelDir);

    // 加载配置
    fs::path configPath = modelPath / "blip_vqa_config.json";
    if (fs::exists(configPath)) {
        loadConfig(configPath.string());
    }

    // 加载视觉编码器
    fs::path visualPath = modelPath / "blip_vqa_visual_encoder.onnx";
    if (fs::exists(visualPath)) {
        try {
#ifdef _WIN32
            std::wstring wPath = visualPath.wstring();
            visualEncoder_ = std::make_unique<Ort::Session>(*env_, wPath.c_str(), sessionOptions_);
#else
            visualEncoder_ = std::make_unique<Ort::Session>(*env_, visualPath.c_str(), sessionOptions_);
#endif
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
            std::cout << "BLIP VQA visual encoder loaded: " << visualPath << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to load VQA visual encoder: " << e.what() << std::endl;
        }
    }

    // 加载文本编码器
    fs::path textEncoderPath = modelPath / "blip_vqa_text_encoder.onnx";
    if (fs::exists(textEncoderPath)) {
        try {
#ifdef _WIN32
            std::wstring wPath = textEncoderPath.wstring();
            textEncoder_ = std::make_unique<Ort::Session>(*env_, wPath.c_str(), sessionOptions_);
#else
            textEncoder_ = std::make_unique<Ort::Session>(*env_, textEncoderPath.c_str(), sessionOptions_);
#endif
            Ort::AllocatorWithDefaultOptions allocator;
            for (size_t i = 0; i < textEncoder_->GetInputCount(); ++i) {
                auto name = textEncoder_->GetInputNameAllocated(i, allocator);
                textEncoderInputNames_.push_back(name.get());
            }
            for (size_t i = 0; i < textEncoder_->GetOutputCount(); ++i) {
                auto name = textEncoder_->GetOutputNameAllocated(i, allocator);
                textEncoderOutputNames_.push_back(name.get());
            }
            textEncoderLoaded_ = true;
            std::cout << "BLIP VQA text encoder loaded: " << textEncoderPath << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to load VQA text encoder: " << e.what() << std::endl;
        }
    }

    // 加载文本解码器
    fs::path decoderPath = modelPath / "blip_vqa_text_decoder.onnx";
    if (fs::exists(decoderPath)) {
        try {
#ifdef _WIN32
            std::wstring wPath = decoderPath.wstring();
            textDecoder_ = std::make_unique<Ort::Session>(*env_, wPath.c_str(), sessionOptions_);
#else
            textDecoder_ = std::make_unique<Ort::Session>(*env_, decoderPath.c_str(), sessionOptions_);
#endif
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
            std::cout << "BLIP VQA text decoder loaded: " << decoderPath << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to load VQA text decoder: " << e.what() << std::endl;
        }
    }

    // 加载词表
    fs::path vocabPath = modelPath / "tokenizer" / "vocab.txt";
    if (fs::exists(vocabPath)) {
        loadVocab(vocabPath.string());
    }
}

bool VqaModel::loadConfig(const std::string& configPath) {
    std::ifstream file(configPath);
    if (!file.is_open()) {
        return false;
    }

    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());

    auto getValue = [&content](const std::string& key) -> std::string {
        size_t pos = content.find("\"" + key + "\"");
        if (pos == std::string::npos) return "";
        pos = content.find(":", pos);
        if (pos == std::string::npos) return "";
        pos++;
        while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t')) pos++;
        size_t end = content.find_first_of(",}\n", pos);
        std::string value = content.substr(pos, end - pos);
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
    config_.maxQuestionLength = getInt("max_question_length", 32);
    config_.maxAnswerLength = getInt("max_answer_length", 16);
    config_.vocabSize = getInt("vocab_size", 21128);
    config_.hiddenSize = getInt("hidden_size", 768);
    config_.bosTokenId = getInt("bos_token_id", 101);
    config_.eosTokenId = getInt("eos_token_id", 102);
    config_.padTokenId = getInt("pad_token_id", 0);

    std::cout << "BLIP VQA config loaded: image_size=" << config_.imageSize
              << ", vocab_size=" << config_.vocabSize << std::endl;
    return true;
}

bool VqaModel::loadVocab(const std::string& vocabPath) {
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
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ')) {
            line.pop_back();
        }
        id2token_.push_back(line);
        token2id_[line] = id;
        id++;
    }

    std::cout << "VQA Vocab loaded: " << id2token_.size() << " tokens" << std::endl;
    return true;
}

std::vector<float> VqaModel::preprocessImage(const cv::Mat& image) {
    cv::Mat processed;

    cv::resize(image, processed, cv::Size(config_.imageSize, config_.imageSize));
    cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);
    processed.convertTo(processed, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(processed, channels);

    for (int c = 0; c < 3; ++c) {
        channels[c] = (channels[c] - config_.imageMean[c]) / config_.imageStd[c];
    }

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

std::vector<int64_t> VqaModel::tokenize(const std::string& text) {
    std::vector<int64_t> tokens;
    tokens.push_back(config_.bosTokenId);  // [CLS]

    // 简单的字符级分词 (适用于中文)
    size_t i = 0;
    while (i < text.size() && tokens.size() < static_cast<size_t>(config_.maxQuestionLength - 1)) {
        // UTF-8 字符处理
        unsigned char c = text[i];
        size_t charLen = 1;
        if ((c & 0x80) == 0) {
            charLen = 1;
        } else if ((c & 0xE0) == 0xC0) {
            charLen = 2;
        } else if ((c & 0xF0) == 0xE0) {
            charLen = 3;
        } else if ((c & 0xF8) == 0xF0) {
            charLen = 4;
        }

        std::string ch = text.substr(i, charLen);
        auto it = token2id_.find(ch);
        if (it != token2id_.end()) {
            tokens.push_back(it->second);
        } else {
            // [UNK] token
            auto unkIt = token2id_.find("[UNK]");
            if (unkIt != token2id_.end()) {
                tokens.push_back(unkIt->second);
            }
        }
        i += charLen;
    }

    tokens.push_back(config_.eosTokenId);  // [SEP]

    // Padding
    while (tokens.size() < static_cast<size_t>(config_.maxQuestionLength)) {
        tokens.push_back(config_.padTokenId);
    }

    return tokens;
}

std::vector<float> VqaModel::encodeImage(const cv::Mat& image) {
    if (!visualEncoderLoaded_) {
        throw std::runtime_error("VQA Visual encoder not loaded");
    }

    std::vector<float> inputData = preprocessImage(image);
    std::vector<int64_t> inputShape = {1, 3, config_.imageSize, config_.imageSize};

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo_,
        inputData.data(),
        inputData.size(),
        inputShape.data(),
        inputShape.size()
    );

    std::vector<const char*> inputNames;
    for (const auto& name : visualInputNames_) {
        inputNames.push_back(name.c_str());
    }
    std::vector<const char*> outputNames;
    for (const auto& name : visualOutputNames_) {
        outputNames.push_back(name.c_str());
    }

    auto outputs = visualEncoder_->Run(
        Ort::RunOptions{nullptr},
        inputNames.data(),
        &inputTensor,
        1,
        outputNames.data(),
        outputNames.size()
    );

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

std::vector<float> VqaModel::encodeQuestion(const std::vector<int64_t>& tokens,
                                             const std::vector<float>& imageEmbeds) {
    if (!textEncoderLoaded_) {
        throw std::runtime_error("VQA Text encoder not loaded");
    }

    int64_t batchSize = 1;
    int64_t seqLen = static_cast<int64_t>(tokens.size());
    int64_t encoderSeqLen = imageEmbeds.size() / config_.hiddenSize;

    std::vector<int64_t> inputIdsShape = {batchSize, seqLen};
    std::vector<int64_t> attentionMask(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        attentionMask[i] = (tokens[i] != config_.padTokenId) ? 1 : 0;
    }
    std::vector<int64_t> encoderShape = {batchSize, encoderSeqLen, static_cast<int64_t>(config_.hiddenSize)};

    Ort::Value inputIdsTensor = Ort::Value::CreateTensor<int64_t>(
        memoryInfo_,
        const_cast<int64_t*>(tokens.data()),
        tokens.size(),
        inputIdsShape.data(),
        inputIdsShape.size()
    );

    Ort::Value attentionTensor = Ort::Value::CreateTensor<int64_t>(
        memoryInfo_,
        attentionMask.data(),
        attentionMask.size(),
        inputIdsShape.data(),
        inputIdsShape.size()
    );

    Ort::Value encoderTensor = Ort::Value::CreateTensor<float>(
        memoryInfo_,
        const_cast<float*>(imageEmbeds.data()),
        imageEmbeds.size(),
        encoderShape.data(),
        encoderShape.size()
    );

    std::vector<const char*> inputNames;
    for (const auto& name : textEncoderInputNames_) {
        inputNames.push_back(name.c_str());
    }
    std::vector<const char*> outputNames;
    for (const auto& name : textEncoderOutputNames_) {
        outputNames.push_back(name.c_str());
    }

    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(inputIdsTensor));
    inputs.push_back(std::move(attentionTensor));
    inputs.push_back(std::move(encoderTensor));

    auto outputs = textEncoder_->Run(
        Ort::RunOptions{nullptr},
        inputNames.data(),
        inputs.data(),
        inputs.size(),
        outputNames.data(),
        outputNames.size()
    );

    const Ort::Value& output = outputs[0];
    auto typeInfo = output.GetTensorTypeAndShapeInfo();
    size_t totalSize = 1;
    for (auto dim : typeInfo.GetShape()) {
        totalSize *= dim;
    }

    const float* data = output.GetTensorData<float>();
    return std::vector<float>(data, data + totalSize);
}

std::vector<int64_t> VqaModel::greedyDecode(const std::vector<float>& questionEmbeds, int maxLength) {
    if (!textDecoderLoaded_) {
        throw std::runtime_error("VQA Text decoder not loaded");
    }

    std::vector<int64_t> generatedTokens;
    generatedTokens.push_back(config_.bosTokenId);

    int64_t batchSize = 1;
    int64_t encoderSeqLen = questionEmbeds.size() / config_.hiddenSize;
    std::vector<int64_t> encoderShape = {batchSize, encoderSeqLen, static_cast<int64_t>(config_.hiddenSize)};

    for (int step = 0; step < maxLength; ++step) {
        std::vector<int64_t> inputIds = generatedTokens;
        std::vector<int64_t> inputIdsShape = {batchSize, static_cast<int64_t>(inputIds.size())};

        Ort::Value inputIdsTensor = Ort::Value::CreateTensor<int64_t>(
            memoryInfo_,
            inputIds.data(),
            inputIds.size(),
            inputIdsShape.data(),
            inputIdsShape.size()
        );

        Ort::Value encoderTensor = Ort::Value::CreateTensor<float>(
            memoryInfo_,
            const_cast<float*>(questionEmbeds.data()),
            questionEmbeds.size(),
            encoderShape.data(),
            encoderShape.size()
        );

        std::vector<const char*> inputNames;
        for (const auto& name : decoderInputNames_) {
            inputNames.push_back(name.c_str());
        }
        std::vector<const char*> outputNames;
        for (const auto& name : decoderOutputNames_) {
            outputNames.push_back(name.c_str());
        }

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

        const Ort::Value& logits = outputs[0];
        auto logitsInfo = logits.GetTensorTypeAndShapeInfo();
        auto logitsShape = logitsInfo.GetShape();

        int64_t seqLen = logitsShape[1];
        int64_t vocabSize = logitsShape[2];

        const float* logitsData = logits.GetTensorData<float>();
        const float* lastLogits = logitsData + (seqLen - 1) * vocabSize;

        int64_t nextToken = 0;
        float maxLogit = lastLogits[0];
        for (int64_t i = 1; i < vocabSize; ++i) {
            if (lastLogits[i] > maxLogit) {
                maxLogit = lastLogits[i];
                nextToken = i;
            }
        }

        if (nextToken == config_.eosTokenId) {
            break;
        }

        generatedTokens.push_back(nextToken);
    }

    return generatedTokens;
}

std::string VqaModel::decodeTokens(const std::vector<int64_t>& tokens) {
    if (id2token_.empty()) {
        std::ostringstream oss;
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i > 0) oss << " ";
            oss << tokens[i];
        }
        return oss.str();
    }

    std::string result;
    for (int64_t token : tokens) {
        if (token == config_.bosTokenId || token == config_.eosTokenId || token == config_.padTokenId) {
            continue;
        }

        if (token >= 0 && token < static_cast<int64_t>(id2token_.size())) {
            std::string tokenStr = id2token_[token];

            if (tokenStr.size() >= 2 && tokenStr[0] == '#' && tokenStr[1] == '#') {
                result += tokenStr.substr(2);
            } else if (tokenStr == "[UNK]") {
                result += "?";
            } else {
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

std::string VqaModel::answer(const cv::Mat& image, const std::string& question) {
    if (!loaded()) {
        throw std::runtime_error("BLIP VQA model not loaded. Please place model files in assets/models/blip_vqa/");
    }

    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    if (question.empty()) {
        throw std::runtime_error("Question is empty");
    }

    // 1. 编码图像
    std::vector<float> imageEmbeds = encodeImage(image);

    // 2. 分词问题
    std::vector<int64_t> questionTokens = tokenize(question);

    // 3. 编码问题 (结合图像特征)
    std::vector<float> questionEmbeds = encodeQuestion(questionTokens, imageEmbeds);

    // 4. 解码生成答案
    std::vector<int64_t> answerTokens = greedyDecode(questionEmbeds, config_.maxAnswerLength);

    // 5. 转换为文本
    return decodeTokens(answerTokens);
}

} // namespace core
} // namespace vindex
