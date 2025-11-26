#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

namespace vindex {
namespace core {

/**
 * @brief CLIP图像预处理器
 *
 * 将输入图像预处理为CLIP模型所需的格式：
 * - Resize到224x224
 * - 归一化：mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
 * - 转换为NCHW格式的float数组
 */
class ImagePreprocessor {
public:
    ImagePreprocessor();
    ~ImagePreprocessor() = default;

    /**
     * @brief 从文件路径加载并预处理图像
     * @param imagePath 图像文件路径
     * @return 预处理后的float向量 (1, 3, 224, 224)
     */
    std::vector<float> preprocess(const std::string& imagePath);

    /**
     * @brief 从cv::Mat预处理图像
     * @param image OpenCV图像矩阵
     * @return 预处理后的float向量 (1, 3, 224, 224)
     */
    std::vector<float> preprocess(const cv::Mat& image);

    /**
     * @brief 批量预处理图像
     * @param images 图像矩阵列表
     * @return 预处理后的float向量 (batch_size, 3, 224, 224)
     */
    std::vector<float> preprocessBatch(const std::vector<cv::Mat>& images);

    /**
     * @brief 获取单张图像的输入尺寸（用于ONNX推理）
     * @return [batch, channels, height, width]
     */
    std::vector<int64_t> getInputShape() const {
        return {1, 3, inputSize_, inputSize_};
    }

    /**
     * @brief 获取批量图像的输入尺寸
     */
    std::vector<int64_t> getBatchInputShape(size_t batchSize) const {
        return {static_cast<int64_t>(batchSize), 3, inputSize_, inputSize_};
    }

    int getInputSize() const { return inputSize_; }

private:
    /**
     * @brief 核心预处理逻辑
     * @param image 输入图像（BGR格式）
     * @param output 输出缓冲区
     * @param offset 输出缓冲区偏移量（用于批处理）
     */
    void preprocessInternal(const cv::Mat& image, std::vector<float>& output, size_t offset);

    /**
     * @brief 验证并转换图像格式
     */
    cv::Mat validateAndConvert(const cv::Mat& image);

private:
    int inputSize_;                    // 输入图像尺寸 (224)
    std::vector<float> mean_;          // RGB均值
    std::vector<float> std_;           // RGB标准差
};

} // namespace core
} // namespace vindex
