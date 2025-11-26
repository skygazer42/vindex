#include "image_preprocessor.h"
#include <stdexcept>
#include <cstring>

namespace vindex {
namespace core {

ImagePreprocessor::ImagePreprocessor()
    : inputSize_(224)
    , mean_{0.48145466f, 0.4578275f, 0.40821073f}  // CLIP默认RGB均值
    , std_{0.26862954f, 0.26130258f, 0.27577711f}   // CLIP默认RGB标准差
{
}

std::vector<float> ImagePreprocessor::preprocess(const std::string& imagePath) {
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + imagePath);
    }
    return preprocess(image);
}

std::vector<float> ImagePreprocessor::preprocess(const cv::Mat& image) {
    const size_t totalSize = 3 * inputSize_ * inputSize_;
    std::vector<float> output(totalSize);
    preprocessInternal(image, output, 0);
    return output;
}

std::vector<float> ImagePreprocessor::preprocessBatch(const std::vector<cv::Mat>& images) {
    if (images.empty()) {
        throw std::invalid_argument("Empty image batch");
    }

    const size_t batchSize = images.size();
    const size_t singleImageSize = 3 * inputSize_ * inputSize_;
    std::vector<float> output(batchSize * singleImageSize);

    for (size_t i = 0; i < batchSize; ++i) {
        preprocessInternal(images[i], output, i * singleImageSize);
    }

    return output;
}

void ImagePreprocessor::preprocessInternal(const cv::Mat& image,
                                           std::vector<float>& output,
                                           size_t offset) {
    // 1. 验证并转换格式
    cv::Mat validImage = validateAndConvert(image);

    // 2. Resize到目标尺寸 (224x224)
    cv::Mat resized;
    if (validImage.rows != inputSize_ || validImage.cols != inputSize_) {
        cv::resize(validImage, resized, cv::Size(inputSize_, inputSize_), 0, 0, cv::INTER_LINEAR);
    } else {
        resized = validImage;
    }

    // 3. BGR转RGB
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    // 4. 转换为float并归一化到[0,1]
    cv::Mat floatImage;
    rgb.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);

    // 5. 归一化并转换为CHW格式
    // OpenCV格式: HWC (Height x Width x Channels)
    // ONNX格式: CHW (Channels x Height x Width)
    const int H = inputSize_;
    const int W = inputSize_;
    const int C = 3;

    float* outputPtr = output.data() + offset;

    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                // HWC -> CHW 索引转换
                float pixelValue = floatImage.at<cv::Vec3f>(h, w)[c];

                // 标准化: (pixel - mean) / std
                float normalized = (pixelValue - mean_[c]) / std_[c];

                // 存储到CHW格式
                outputPtr[c * H * W + h * W + w] = normalized;
            }
        }
    }
}

cv::Mat ImagePreprocessor::validateAndConvert(const cv::Mat& image) {
    if (image.empty()) {
        throw std::invalid_argument("Input image is empty");
    }

    // 如果是灰度图，转换为BGR
    if (image.channels() == 1) {
        cv::Mat bgr;
        cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
        return bgr;
    }

    // 如果是RGBA，转换为BGR
    if (image.channels() == 4) {
        cv::Mat bgr;
        cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
        return bgr;
    }

    // 如果已经是BGR，直接返回
    if (image.channels() == 3) {
        return image.clone();
    }

    throw std::invalid_argument("Unsupported image format");
}

} // namespace core
} // namespace vindex
