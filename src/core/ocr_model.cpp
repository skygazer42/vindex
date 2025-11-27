#include "ocr_model.h"
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

namespace fs = std::filesystem;

namespace vindex {
namespace core {

OcrModel::OcrModel(Ort::Env& env, const std::string& modelDir, int numThreads)
    : env_(&env)
    , memoryInfo_(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault))
{
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    sessionOptions_.SetIntraOpNumThreads(numThreads);

    if (modelDir.empty() || !fs::exists(modelDir)) {
        std::cerr << "OCR model directory not found: " << modelDir << std::endl;
        return;
    }

    fs::path modelPath(modelDir);

    // 加载配置
    fs::path configPath = modelPath / "ocr_config.json";
    if (fs::exists(configPath)) {
        loadConfig(configPath.string());
    }

    // 加载检测模型
    fs::path detPath = modelPath / "ch_PP-OCRv4_det_infer.onnx";
    if (fs::exists(detPath)) {
        try {
#ifdef _WIN32
            std::wstring wPath = detPath.wstring();
            detModel_ = std::make_unique<Ort::Session>(*env_, wPath.c_str(), sessionOptions_);
#else
            detModel_ = std::make_unique<Ort::Session>(*env_, detPath.c_str(), sessionOptions_);
#endif
            Ort::AllocatorWithDefaultOptions allocator;
            for (size_t i = 0; i < detModel_->GetInputCount(); ++i) {
                auto name = detModel_->GetInputNameAllocated(i, allocator);
                detInputNames_.push_back(name.get());
            }
            for (size_t i = 0; i < detModel_->GetOutputCount(); ++i) {
                auto name = detModel_->GetOutputNameAllocated(i, allocator);
                detOutputNames_.push_back(name.get());
            }
            detModelLoaded_ = true;
            std::cout << "OCR detection model loaded: " << detPath << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to load OCR detection model: " << e.what() << std::endl;
        }
    }

    // 加载识别模型
    fs::path recPath = modelPath / "ch_PP-OCRv4_rec_infer.onnx";
    if (fs::exists(recPath)) {
        try {
#ifdef _WIN32
            std::wstring wPath = recPath.wstring();
            recModel_ = std::make_unique<Ort::Session>(*env_, wPath.c_str(), sessionOptions_);
#else
            recModel_ = std::make_unique<Ort::Session>(*env_, recPath.c_str(), sessionOptions_);
#endif
            Ort::AllocatorWithDefaultOptions allocator;
            for (size_t i = 0; i < recModel_->GetInputCount(); ++i) {
                auto name = recModel_->GetInputNameAllocated(i, allocator);
                recInputNames_.push_back(name.get());
            }
            for (size_t i = 0; i < recModel_->GetOutputCount(); ++i) {
                auto name = recModel_->GetOutputNameAllocated(i, allocator);
                recOutputNames_.push_back(name.get());
            }
            recModelLoaded_ = true;
            std::cout << "OCR recognition model loaded: " << recPath << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to load OCR recognition model: " << e.what() << std::endl;
        }
    }

    // 加载字典
    fs::path dictPath = modelPath / "ppocr_keys_v1.txt";
    if (fs::exists(dictPath)) {
        loadDict(dictPath.string());
    }
}

bool OcrModel::loadConfig(const std::string& configPath) {
    std::ifstream file(configPath);
    if (!file.is_open()) {
        return false;
    }

    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());

    auto getFloat = [&content](const std::string& key, float defaultVal) -> float {
        size_t pos = content.find("\"" + key + "\"");
        if (pos == std::string::npos) return defaultVal;
        pos = content.find(":", pos);
        if (pos == std::string::npos) return defaultVal;
        pos++;
        while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t')) pos++;
        size_t end = content.find_first_of(",}\n", pos);
        std::string value = content.substr(pos, end - pos);
        return value.empty() ? defaultVal : std::stof(value);
    };

    auto getInt = [&content](const std::string& key, int defaultVal) -> int {
        size_t pos = content.find("\"" + key + "\"");
        if (pos == std::string::npos) return defaultVal;
        pos = content.find(":", pos);
        if (pos == std::string::npos) return defaultVal;
        pos++;
        while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t')) pos++;
        size_t end = content.find_first_of(",}\n", pos);
        std::string value = content.substr(pos, end - pos);
        return value.empty() ? defaultVal : std::stoi(value);
    };

    config_.detDbThresh = getFloat("det_db_thresh", 0.3f);
    config_.detDbBoxThresh = getFloat("det_db_box_thresh", 0.5f);
    config_.detDbUnclipRatio = getFloat("det_db_unclip_ratio", 1.6f);
    config_.recImgHeight = getInt("rec_img_height", 48);
    config_.recImgWidth = getInt("rec_img_width", 320);
    config_.maxSideLen = getInt("max_side_len", 960);

    std::cout << "OCR config loaded" << std::endl;
    return true;
}

bool OcrModel::loadDict(const std::string& dictPath) {
    std::ifstream file(dictPath);
    if (!file.is_open()) {
        std::cerr << "Failed to open dict file: " << dictPath << std::endl;
        return false;
    }

    dict_.clear();
    dict_.push_back(" ");  // 空白符作为第一个字符 (CTC blank)

    std::string line;
    while (std::getline(file, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
            line.pop_back();
        }
        if (!line.empty()) {
            dict_.push_back(line);
        }
    }

    dict_.push_back(" ");  // 末尾空格

    std::cout << "OCR dict loaded: " << dict_.size() << " characters" << std::endl;
    return true;
}

cv::Mat OcrModel::preprocessForDet(const cv::Mat& image, float& ratioH, float& ratioW) {
    int h = image.rows;
    int w = image.cols;

    // 计算缩放比例
    float ratio = 1.0f;
    int maxWH = std::max(h, w);
    if (maxWH > config_.maxSideLen) {
        ratio = static_cast<float>(config_.maxSideLen) / maxWH;
    }

    int newH = static_cast<int>(h * ratio);
    int newW = static_cast<int>(w * ratio);

    // 确保尺寸是32的倍数
    newH = std::max(32, (newH / 32) * 32);
    newW = std::max(32, (newW / 32) * 32);

    ratioH = static_cast<float>(h) / newH;
    ratioW = static_cast<float>(w) / newW;

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(newW, newH));

    // 转换为 RGB
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    // 归一化
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    // 标准化
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std = {0.229f, 0.224f, 0.225f};
    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);
    for (int i = 0; i < 3; ++i) {
        channels[i] = (channels[i] - mean[i]) / std[i];
    }
    cv::merge(channels, rgb);

    return rgb;
}

cv::Mat OcrModel::preprocessForRec(const cv::Mat& image) {
    int h = image.rows;
    int w = image.cols;

    // 计算目标尺寸
    float ratio = static_cast<float>(config_.recImgHeight) / h;
    int newW = static_cast<int>(w * ratio);

    // 限制最大宽度
    if (newW > config_.recImgWidth) {
        newW = config_.recImgWidth;
    }

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(newW, config_.recImgHeight));

    // 如果宽度不够，右侧补零
    cv::Mat padded = cv::Mat::zeros(config_.recImgHeight, config_.recImgWidth, CV_8UC3);
    resized.copyTo(padded(cv::Rect(0, 0, resized.cols, resized.rows)));

    // 转换为 RGB
    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

    // 归一化到 [-1, 1]
    rgb.convertTo(rgb, CV_32F, 1.0 / 127.5, -1.0);

    return rgb;
}

std::vector<std::vector<cv::Point2f>> OcrModel::postprocessDet(
    const float* data, int h, int w, float ratioH, float ratioW) {

    std::vector<std::vector<cv::Point2f>> boxes;

    // 创建二值图
    cv::Mat bitmap(h, w, CV_8UC1);
    for (int i = 0; i < h * w; ++i) {
        bitmap.data[i] = data[i] > config_.detDbThresh ? 255 : 0;
    }

    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bitmap, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours) {
        if (contour.size() < 4) continue;

        // 计算最小外接矩形
        cv::RotatedRect rect = cv::minAreaRect(contour);

        // 过滤太小的区域
        float shortSide = std::min(rect.size.width, rect.size.height);
        if (shortSide < 3) continue;

        // 计算得分
        float score = 0;
        cv::Mat mask = cv::Mat::zeros(h, w, CV_8UC1);
        cv::drawContours(mask, std::vector<std::vector<cv::Point>>{contour}, 0, cv::Scalar(255), -1);
        int count = 0;
        for (int i = 0; i < h * w; ++i) {
            if (mask.data[i] > 0) {
                score += data[i];
                count++;
            }
        }
        if (count > 0) score /= count;

        if (score < config_.detDbBoxThresh) continue;

        // 获取矩形的四个角点
        cv::Point2f points[4];
        rect.points(points);

        // 扩展矩形
        float expandRatio = config_.detDbUnclipRatio;
        cv::Point2f center = rect.center;
        std::vector<cv::Point2f> expandedBox(4);
        for (int i = 0; i < 4; ++i) {
            float dx = points[i].x - center.x;
            float dy = points[i].y - center.y;
            expandedBox[i].x = (center.x + dx * expandRatio) * ratioW;
            expandedBox[i].y = (center.y + dy * expandRatio) * ratioH;
        }

        boxes.push_back(expandedBox);
    }

    return boxes;
}

cv::Mat OcrModel::cropTextRegion(const cv::Mat& image, const std::vector<cv::Point2f>& box) {
    // 计算目标矩形
    float width = std::max(
        std::sqrt(std::pow(box[0].x - box[1].x, 2) + std::pow(box[0].y - box[1].y, 2)),
        std::sqrt(std::pow(box[2].x - box[3].x, 2) + std::pow(box[2].y - box[3].y, 2))
    );
    float height = std::max(
        std::sqrt(std::pow(box[0].x - box[3].x, 2) + std::pow(box[0].y - box[3].y, 2)),
        std::sqrt(std::pow(box[1].x - box[2].x, 2) + std::pow(box[1].y - box[2].y, 2))
    );

    // 源点和目标点
    std::vector<cv::Point2f> srcPoints = box;
    std::vector<cv::Point2f> dstPoints = {
        cv::Point2f(0, 0),
        cv::Point2f(width, 0),
        cv::Point2f(width, height),
        cv::Point2f(0, height)
    };

    // 透视变换
    cv::Mat transform = cv::getPerspectiveTransform(srcPoints, dstPoints);
    cv::Mat cropped;
    cv::warpPerspective(image, cropped, transform, cv::Size(static_cast<int>(width), static_cast<int>(height)));

    return cropped;
}

void OcrModel::sortBoxes(std::vector<std::vector<cv::Point2f>>& boxes) {
    // 按照 y 坐标排序，然后按 x 坐标排序
    std::sort(boxes.begin(), boxes.end(), [](const auto& a, const auto& b) {
        float ay = (a[0].y + a[1].y + a[2].y + a[3].y) / 4;
        float by = (b[0].y + b[1].y + b[2].y + b[3].y) / 4;
        float ax = (a[0].x + a[1].x + a[2].x + a[3].x) / 4;
        float bx = (b[0].x + b[1].x + b[2].x + b[3].x) / 4;

        // 如果 y 坐标差距较小，则按 x 排序
        if (std::abs(ay - by) < 10) {
            return ax < bx;
        }
        return ay < by;
    });
}

std::vector<std::vector<cv::Point2f>> OcrModel::detect(const cv::Mat& image) {
    if (!detModelLoaded_) {
        return {};
    }

    float ratioH, ratioW;
    cv::Mat processed = preprocessForDet(image, ratioH, ratioW);

    int h = processed.rows;
    int w = processed.cols;

    // 转换为 NCHW 格式
    std::vector<float> inputData(3 * h * w);
    std::vector<cv::Mat> channels(3);
    cv::split(processed, channels);
    for (int c = 0; c < 3; ++c) {
        std::memcpy(inputData.data() + c * h * w, channels[c].data, h * w * sizeof(float));
    }

    std::vector<int64_t> inputShape = {1, 3, h, w};
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo_, inputData.data(), inputData.size(),
        inputShape.data(), inputShape.size()
    );

    std::vector<const char*> inputNames = {detInputNames_[0].c_str()};
    std::vector<const char*> outputNames = {detOutputNames_[0].c_str()};

    auto outputs = detModel_->Run(
        Ort::RunOptions{nullptr},
        inputNames.data(), &inputTensor, 1,
        outputNames.data(), outputNames.size()
    );

    const Ort::Value& output = outputs[0];
    auto outputInfo = output.GetTensorTypeAndShapeInfo();
    auto outputShape = outputInfo.GetShape();

    int outH = static_cast<int>(outputShape[2]);
    int outW = static_cast<int>(outputShape[3]);
    const float* outputData = output.GetTensorData<float>();

    auto boxes = postprocessDet(outputData, outH, outW, ratioH, ratioW);
    sortBoxes(boxes);

    return boxes;
}

std::pair<std::string, float> OcrModel::recognizeOne(const cv::Mat& image) {
    if (!recModelLoaded_ || dict_.empty()) {
        return {"", 0.0f};
    }

    cv::Mat processed = preprocessForRec(image);

    int h = processed.rows;
    int w = processed.cols;

    // 转换为 NCHW 格式
    std::vector<float> inputData(3 * h * w);
    std::vector<cv::Mat> channels(3);
    cv::split(processed, channels);
    for (int c = 0; c < 3; ++c) {
        std::memcpy(inputData.data() + c * h * w, channels[c].data, h * w * sizeof(float));
    }

    std::vector<int64_t> inputShape = {1, 3, h, w};
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo_, inputData.data(), inputData.size(),
        inputShape.data(), inputShape.size()
    );

    std::vector<const char*> inputNames = {recInputNames_[0].c_str()};
    std::vector<const char*> outputNames = {recOutputNames_[0].c_str()};

    auto outputs = recModel_->Run(
        Ort::RunOptions{nullptr},
        inputNames.data(), &inputTensor, 1,
        outputNames.data(), outputNames.size()
    );

    const Ort::Value& output = outputs[0];
    auto outputInfo = output.GetTensorTypeAndShapeInfo();
    auto outputShape = outputInfo.GetShape();

    int seqLen = static_cast<int>(outputShape[1]);
    int numClasses = static_cast<int>(outputShape[2]);
    const float* outputData = output.GetTensorData<float>();

    // CTC 解码
    std::string text;
    float totalConf = 0;
    int confCount = 0;
    int lastIdx = 0;

    for (int t = 0; t < seqLen; ++t) {
        const float* probs = outputData + t * numClasses;

        // 找最大概率的字符
        int maxIdx = 0;
        float maxProb = probs[0];
        for (int c = 1; c < numClasses; ++c) {
            if (probs[c] > maxProb) {
                maxProb = probs[c];
                maxIdx = c;
            }
        }

        // CTC: 去重和跳过 blank
        if (maxIdx != 0 && maxIdx != lastIdx) {
            if (maxIdx < static_cast<int>(dict_.size())) {
                text += dict_[maxIdx];
                totalConf += maxProb;
                confCount++;
            }
        }
        lastIdx = maxIdx;
    }

    float avgConf = confCount > 0 ? totalConf / confCount : 0;
    return {text, avgConf};
}

std::vector<OCRResult> OcrModel::recognize(const cv::Mat& image) {
    if (!loaded()) {
        throw std::runtime_error("OCR model not loaded. Please place model files in assets/models/ocr/");
    }

    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    std::vector<OCRResult> results;

    // 1. 检测文字区域
    auto boxes = detect(image);

    // 2. 识别每个区域
    for (const auto& box : boxes) {
        // 裁剪文字区域
        cv::Mat cropped = cropTextRegion(image, box);
        if (cropped.empty() || cropped.cols < 5 || cropped.rows < 5) {
            continue;
        }

        // 识别文字
        auto [text, confidence] = recognizeOne(cropped);

        if (!text.empty()) {
            OCRResult result;
            result.text = text;
            result.box = box;
            result.confidence = confidence;
            results.push_back(result);
        }
    }

    return results;
}

std::string OcrModel::recognizeText(const cv::Mat& image) {
    auto results = recognize(image);

    std::string text;
    for (const auto& result : results) {
        if (!text.empty()) {
            text += "\n";
        }
        text += result.text;
    }

    return text;
}

} // namespace core
} // namespace vindex
