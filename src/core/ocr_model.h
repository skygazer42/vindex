#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>

namespace vindex {
namespace core {

/**
 * @brief OCR 文字识别结果
 */
struct OCRResult {
    std::string text;           // 识别的文字
    std::vector<cv::Point2f> box;  // 文字区域的四个角点
    float confidence;           // 置信度
};

/**
 * @brief PP-OCRv4 文字识别模型
 *
 * 支持中文文字检测和识别
 * 使用检测模型定位文字区域，识别模型识别文字内容
 */
class OcrModel {
public:
    struct Config {
        float detDbThresh = 0.3f;       // 检测阈值
        float detDbBoxThresh = 0.5f;    // 检测框阈值
        float detDbUnclipRatio = 1.6f;  // 检测框扩展比例
        int recImgHeight = 48;          // 识别输入高度
        int recImgWidth = 320;          // 识别输入宽度
        int maxSideLen = 960;           // 最大边长
    };

    OcrModel(Ort::Env& env, const std::string& modelDir, int numThreads = 4);
    ~OcrModel() = default;

    /**
     * @brief 识别图像中的文字
     * @param image 输入图像
     * @return 识别结果列表
     */
    std::vector<OCRResult> recognize(const cv::Mat& image);

    /**
     * @brief 识别图像中的文字并返回纯文本
     * @param image 输入图像
     * @return 识别的文字（按行排列）
     */
    std::string recognizeText(const cv::Mat& image);

    /**
     * @brief 检查模型是否加载
     */
    bool loaded() const { return detModelLoaded_ && recModelLoaded_; }
    bool detModelLoaded() const { return detModelLoaded_; }
    bool recModelLoaded() const { return recModelLoaded_; }

    /**
     * @brief 加载字典
     */
    bool loadDict(const std::string& dictPath);

    /**
     * @brief 加载配置
     */
    bool loadConfig(const std::string& configPath);

    const Config& config() const { return config_; }

private:
    // 检测文字区域
    std::vector<std::vector<cv::Point2f>> detect(const cv::Mat& image);

    // 识别单个文字区域
    std::pair<std::string, float> recognizeOne(const cv::Mat& image);

    // 图像预处理
    cv::Mat preprocessForDet(const cv::Mat& image, float& ratioH, float& ratioW);
    cv::Mat preprocessForRec(const cv::Mat& image);

    // 后处理
    std::vector<std::vector<cv::Point2f>> postprocessDet(const float* data, int h, int w,
                                                          float ratioH, float ratioW);

    // 从文字区域裁剪图像
    cv::Mat cropTextRegion(const cv::Mat& image, const std::vector<cv::Point2f>& box);

    // 按位置排序文字区域
    void sortBoxes(std::vector<std::vector<cv::Point2f>>& boxes);

private:
    Ort::Env* env_;
    Ort::SessionOptions sessionOptions_;

    // 检测模型
    std::unique_ptr<Ort::Session> detModel_;
    std::vector<std::string> detInputNames_;
    std::vector<std::string> detOutputNames_;
    bool detModelLoaded_ = false;

    // 识别模型
    std::unique_ptr<Ort::Session> recModel_;
    std::vector<std::string> recInputNames_;
    std::vector<std::string> recOutputNames_;
    bool recModelLoaded_ = false;

    // 字典
    std::vector<std::string> dict_;

    // 配置
    Config config_;

    // 内存信息
    Ort::MemoryInfo memoryInfo_;
};

} // namespace core
} // namespace vindex
