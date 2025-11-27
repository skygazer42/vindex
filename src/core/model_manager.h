#pragma once

#include "clip_encoder.h"
#include "caption_model.h"
#include "vqa_model.h"
#include "ocr_model.h"
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <string>
#include <mutex>

namespace vindex {
namespace core {

/**
 * @brief 模型管理器（单例）
 *
 * 负责管理应用中所有的模型：
 * - CLIP编码器（图搜图、文搜图）
 * - Caption模型（图像描述）
 * - VQA模型（视觉问答）
 * - OCR模型（文字识别）
 *
 * 使用懒加载模式，仅在需要时加载模型
 */
class ModelManager {
public:
    /**
     * @brief 获取单例实例
     */
    static ModelManager& instance();

    // 禁止拷贝和赋值
    ModelManager(const ModelManager&) = delete;
    ModelManager& operator=(const ModelManager&) = delete;

    // ==================== 配置 ====================

    /**
     * @brief 设置模型根目录
     * @param basePath 模型文件所在的根目录
     */
    void setModelPath(const std::string& basePath);

    /**
     * @brief 设置词表路径
     */
    void setVocabPath(const std::string& vocabPath);

    /**
     * @brief 设置特征向量维度
     */
    void setEmbeddingDim(int dim);

    // ==================== 模型访问 ====================

    /**
     * @brief 获取CLIP编码器（懒加载）
     * @return CLIP编码器引用
     */
    ClipEncoder& clipEncoder();

    /**
     * @brief 检查CLIP编码器是否已加载
     */
    bool hasClipEncoder() const;

    /**
     * @brief 获取图生文模型（懒加载）
     */
    CaptionModel& captionModel();
    bool hasCaptionModel() const;

    /**
     * @brief 获取VQA模型（懒加载）
     */
    VqaModel& vqaModel();
    bool hasVqaModel() const;

    /**
     * @brief 获取OCR模型（懒加载）
     */
    OcrModel& ocrModel();
    bool hasOcrModel() const;

    // ==================== 预加载 ====================

    /**
     * @brief 预加载所有模型（可选，用于启动时）
     * @return 是否全部加载成功
     */
    bool preloadAll();

    /**
     * @brief 释放所有模型（释放内存）
     */
    void releaseAll();

    // ==================== 配置获取 ====================

    std::string getModelPath() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return modelPath_;
    }
    std::string getVocabPath() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return vocabPath_;
    }
    int getEmbeddingDim() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return embeddingDim_;
    }

private:
    ModelManager();
    ~ModelManager() = default;

    /**
     * @brief 初始化各模型
     */
    void initializeClipEncoder();
    void initializeCaptionModel();
    void initializeVqaModel();
    void initializeOcrModel();

private:
    // 模型路径配置
    std::string modelPath_;      // 模型根目录
    std::string vocabPath_;      // 词表路径
    int embeddingDim_;           // 特征维度

    // 模型实例
    std::unique_ptr<ClipEncoder> clipEncoder_;
    std::unique_ptr<CaptionModel> captionModel_;
    std::unique_ptr<VqaModel> vqaModel_;
    std::unique_ptr<OcrModel> ocrModel_;
    Ort::Env env_;

    // 线程安全
    mutable std::mutex mutex_;
};

} // namespace core
} // namespace vindex
