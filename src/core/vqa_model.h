#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace vindex {
namespace core {

/**
 * @brief BLIP VQA 视觉问答模型
 *
 * 支持 Taiyi-BLIP-750M-Chinese 等中文视觉问答模型
 * 使用视觉编码器 + 文本编码器 + 文本解码器的三阶段生成
 */
class VqaModel {
public:
    struct Config {
        int imageSize = 384;
        int maxQuestionLength = 32;
        int maxAnswerLength = 16;
        int vocabSize = 21128;
        int hiddenSize = 768;
        int bosTokenId = 101;   // [CLS]
        int eosTokenId = 102;   // [SEP]
        int padTokenId = 0;     // [PAD]
        std::vector<float> imageMean = {0.48145466f, 0.4578275f, 0.40821073f};
        std::vector<float> imageStd = {0.26862954f, 0.26130258f, 0.27577711f};
    };

    VqaModel(Ort::Env& env, const std::string& modelDir, int numThreads = 4);
    ~VqaModel() = default;

    /**
     * @brief 回答图像问题
     * @param image 输入图像
     * @param question 问题文本
     * @return 生成的答案
     */
    std::string answer(const cv::Mat& image, const std::string& question);

    /**
     * @brief 检查模型是否加载
     */
    bool loaded() const { return visualEncoderLoaded_ && textEncoderLoaded_ && textDecoderLoaded_; }
    bool visualEncoderLoaded() const { return visualEncoderLoaded_; }
    bool textEncoderLoaded() const { return textEncoderLoaded_; }
    bool textDecoderLoaded() const { return textDecoderLoaded_; }

    /**
     * @brief 加载词表
     */
    bool loadVocab(const std::string& vocabPath);

    /**
     * @brief 加载配置
     */
    bool loadConfig(const std::string& configPath);

    const Config& config() const { return config_; }

private:
    // 图像编码
    std::vector<float> encodeImage(const cv::Mat& image);

    // 问题编码
    std::vector<float> encodeQuestion(const std::vector<int64_t>& tokens,
                                       const std::vector<float>& imageEmbeds);

    // 贪心解码生成答案
    std::vector<int64_t> greedyDecode(const std::vector<float>& questionEmbeds, int maxLength);

    // Token ID 转文本
    std::string decodeTokens(const std::vector<int64_t>& tokens);

    // 图像预处理
    std::vector<float> preprocessImage(const cv::Mat& image);

    // 文本分词 (简单实现)
    std::vector<int64_t> tokenize(const std::string& text);

private:
    Ort::Env* env_;
    Ort::SessionOptions sessionOptions_;

    // 视觉编码器
    std::unique_ptr<Ort::Session> visualEncoder_;
    std::vector<std::string> visualInputNames_;
    std::vector<std::string> visualOutputNames_;
    bool visualEncoderLoaded_ = false;

    // 文本编码器 (问题编码)
    std::unique_ptr<Ort::Session> textEncoder_;
    std::vector<std::string> textEncoderInputNames_;
    std::vector<std::string> textEncoderOutputNames_;
    bool textEncoderLoaded_ = false;

    // 文本解码器 (答案生成)
    std::unique_ptr<Ort::Session> textDecoder_;
    std::vector<std::string> decoderInputNames_;
    std::vector<std::string> decoderOutputNames_;
    bool textDecoderLoaded_ = false;

    // 词表
    std::vector<std::string> id2token_;
    std::unordered_map<std::string, int64_t> token2id_;

    // 配置
    Config config_;

    // 内存信息
    Ort::MemoryInfo memoryInfo_;
};

} // namespace core
} // namespace vindex
