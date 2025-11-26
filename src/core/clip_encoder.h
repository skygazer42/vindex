#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>

#include "image_preprocessor.h"
#include "text_tokenizer.h"

namespace vindex {
namespace core {

/**
 * @brief CLIP编码器（视觉 + 文本）
 *
 * 使用ONNX Runtime加载并运行CLIP模型
 * 支持：图像编码、文本编码、图文相似度计算
 */
class ClipEncoder {
public:
    /**
     * @brief 构造函数
     * @param visualModelPath CLIP视觉编码器ONNX模型路径
     * @param textModelPath CLIP文本编码器ONNX模型路径（可选）
     * @param vocabPath BPE词表路径（如果使用文本编码器，必须提供）
     * @param embeddingDim 特征向量维度（默认768，ViT-L/14）
     */
    explicit ClipEncoder(const std::string& visualModelPath,
                        const std::string& textModelPath = "",
                        const std::string& vocabPath = "",
                        int embeddingDim = 768);

    ~ClipEncoder() = default;

    // ==================== 图像编码 ====================

    /**
     * @brief 从文件路径编码图像
     * @param imagePath 图像文件路径
     * @return 归一化的特征向量 (embeddingDim维)
     */
    std::vector<float> encodeImage(const std::string& imagePath);

    /**
     * @brief 从cv::Mat编码图像
     * @param image OpenCV图像矩阵
     * @return 归一化的特征向量
     */
    std::vector<float> encodeImage(const cv::Mat& image);

    /**
     * @brief 批量编码图像
     * @param images 图像矩阵列表
     * @return 特征向量矩阵 (batch_size * embeddingDim)
     */
    std::vector<std::vector<float>> encodeImageBatch(const std::vector<cv::Mat>& images);

    // ==================== 文本编码 ====================

    /**
     * @brief 编码文本
     * @param text 输入文本
     * @return 归一化的特征向量
     */
    std::vector<float> encodeText(const std::string& text);

    /**
     * @brief 批量编码文本
     * @param texts 文本列表
     * @return 特征向量矩阵
     */
    std::vector<std::vector<float>> encodeTextBatch(const std::vector<std::string>& texts);

    // ==================== 相似度计算 ====================

    /**
     * @brief 计算图文相似度分数
     * @param image 图像
     * @param text 文本
     * @return 余弦相似度 (0-1之间，越高越相似)
     */
    float computeSimilarity(const cv::Mat& image, const std::string& text);

    /**
     * @brief 计算两个特征向量的余弦相似度
     * @param features1 特征向量1
     * @param features2 特征向量2
     * @return 余弦相似度
     */
    static float cosineSimilarity(const std::vector<float>& features1,
                                 const std::vector<float>& features2);

    /**
     * @brief L2归一化
     */
    static void normalizeL2(std::vector<float>& features);

    // ==================== 获取器 ====================

    int getEmbeddingDim() const { return embeddingDim_; }
    bool hasTextEncoder() const { return textSession_ != nullptr; }

private:
    /**
     * @brief 运行视觉编码器推理
     */
    std::vector<float> runVisualInference(const std::vector<float>& imageData,
                                          const std::vector<int64_t>& inputShape);

    /**
     * @brief 运行文本编码器推理
     */
    std::vector<float> runTextInference(const std::vector<int64_t>& textTokens);

    /**
     * @brief 初始化ONNX会话
     */
    void initializeSessions(const std::string& visualModelPath,
                           const std::string& textModelPath);

private:
    // ONNX Runtime 环境和会话
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> visualSession_;
    std::unique_ptr<Ort::Session> textSession_;
    Ort::SessionOptions sessionOptions_;
    Ort::MemoryInfo memoryInfo_;

    // 预处理器
    std::unique_ptr<ImagePreprocessor> imagePreprocessor_;
    std::unique_ptr<TextTokenizer> textTokenizer_;

    // 模型参数
    int embeddingDim_;

    // 输入/输出名称（从ONNX模型获取）
    std::vector<const char*> visualInputNames_;
    std::vector<const char*> visualOutputNames_;
    std::vector<const char*> textInputNames_;
    std::vector<const char*> textOutputNames_;

    // 名称存储（保证指针有效性）
    std::vector<std::string> visualInputNamesStorage_;
    std::vector<std::string> visualOutputNamesStorage_;
    std::vector<std::string> textInputNamesStorage_;
    std::vector<std::string> textOutputNamesStorage_;
};

} // namespace core
} // namespace vindex
