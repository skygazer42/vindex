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
 * @brief BLIP 图像描述模型
 *
 * 支持 Taiyi-BLIP-750M-Chinese 等中文图像描述模型
 * 使用视觉编码器 + 文本解码器的两阶段生成
 */
class CaptionModel {
public:
    struct Config {
        int imageSize = 384;
        int maxLength = 64;
        int vocabSize = 21128;  // BERT Chinese vocab size
        int hiddenSize = 768;
        int bosTokenId = 101;   // [CLS]
        int eosTokenId = 102;   // [SEP]
        int padTokenId = 0;     // [PAD]
        std::vector<float> imageMean = {0.48145466f, 0.4578275f, 0.40821073f};
        std::vector<float> imageStd = {0.26862954f, 0.26130258f, 0.27577711f};
    };

    CaptionModel(Ort::Env& env, const std::string& modelDir, int numThreads = 4);
    ~CaptionModel() = default;

    /**
     * @brief 生成图像描述
     * @param image 输入图像
     * @param maxLength 最大生成长度
     * @param numBeams beam search 宽度 (1=贪心解码)
     * @return 生成的描述文本
     */
    std::string generate(const cv::Mat& image, int maxLength = 64, int numBeams = 1);

    /**
     * @brief 检查模型是否加载
     */
    bool loaded() const { return visualEncoderLoaded_ && textDecoderLoaded_; }
    bool visualEncoderLoaded() const { return visualEncoderLoaded_; }
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

    // 贪心解码
    std::vector<int64_t> greedyDecode(const std::vector<float>& imageEmbeds, int maxLength);

    // Beam Search 解码
    std::vector<int64_t> beamSearchDecode(const std::vector<float>& imageEmbeds, int maxLength, int numBeams);

    // Token ID 转文本
    std::string decodeTokens(const std::vector<int64_t>& tokens);

    // 图像预处理
    std::vector<float> preprocessImage(const cv::Mat& image);

private:
    Ort::Env* env_;
    Ort::SessionOptions sessionOptions_;

    // 视觉编码器
    std::unique_ptr<Ort::Session> visualEncoder_;
    std::vector<std::string> visualInputNames_;
    std::vector<std::string> visualOutputNames_;
    bool visualEncoderLoaded_ = false;

    // 文本解码器
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
