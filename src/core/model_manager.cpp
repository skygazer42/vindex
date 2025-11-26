#include "model_manager.h"
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace vindex {
namespace core {

ModelManager::ModelManager()
    : modelPath_("./assets/models")
    , vocabPath_("./assets/vocab/bpe_simple_vocab_16e6.txt")
    , embeddingDim_(768)  // ViT-L/14默认维度
    , env_(ORT_LOGGING_LEVEL_WARNING, "ModelManager")
{
}

ModelManager& ModelManager::instance() {
    static ModelManager instance;
    return instance;
}

// ==================== 配置 ====================

void ModelManager::setModelPath(const std::string& basePath) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!fs::exists(basePath)) {
        std::cerr << "Warning: Model path does not exist: " << basePath << std::endl;
    }

    modelPath_ = basePath;
}

void ModelManager::setVocabPath(const std::string& vocabPath) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!fs::exists(vocabPath)) {
        std::cerr << "Warning: Vocabulary path does not exist: " << vocabPath << std::endl;
    }

    vocabPath_ = vocabPath;
}

void ModelManager::setEmbeddingDim(int dim) {
    std::lock_guard<std::mutex> lock(mutex_);
    embeddingDim_ = dim;
}

// ==================== 模型访问 ====================

ClipEncoder& ModelManager::clipEncoder() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!clipEncoder_) {
        initializeClipEncoder();
    }

    return *clipEncoder_;
}

bool ModelManager::hasClipEncoder() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return clipEncoder_ != nullptr;
}

CaptionModel& ModelManager::captionModel() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!captionModel_) {
        initializeCaptionModel();
    }
    return *captionModel_;
}

bool ModelManager::hasCaptionModel() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return captionModel_ != nullptr;
}

VqaModel& ModelManager::vqaModel() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!vqaModel_) {
        initializeVqaModel();
    }
    return *vqaModel_;
}

bool ModelManager::hasVqaModel() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return vqaModel_ != nullptr;
}

// ==================== 预加载 ====================

bool ModelManager::preloadAll() {
    std::lock_guard<std::mutex> lock(mutex_);

    std::cout << "Preloading all models..." << std::endl;

    try {
        // 加载CLIP编码器
        if (!clipEncoder_) {
            initializeClipEncoder();
        }
        // 可选加载其他模型
        if (!captionModel_) {
            initializeCaptionModel();
        }
        if (!vqaModel_) {
            initializeVqaModel();
        }

        std::cout << "All models loaded successfully!" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Failed to preload models: " << e.what() << std::endl;
        return false;
    }
}

void ModelManager::releaseAll() {
    std::lock_guard<std::mutex> lock(mutex_);

    std::cout << "Releasing all models..." << std::endl;

    clipEncoder_.reset();
    captionModel_.reset();
    vqaModel_.reset();

    std::cout << "All models released!" << std::endl;
}

// ==================== 私有方法 ====================

void ModelManager::initializeClipEncoder() {
    std::cout << "Initializing CLIP encoder..." << std::endl;

    // 构建模型路径
    std::string visualModelPath = (fs::path(modelPath_) / "clip_visual.onnx").string();
    std::string textModelPath = (fs::path(modelPath_) / "clip_text.onnx").string();

    // 检查文件是否存在
    if (!fs::exists(visualModelPath)) {
        throw std::runtime_error("CLIP visual model not found: " + visualModelPath);
    }

    // 文本模型是可选的
    if (!fs::exists(textModelPath)) {
        std::cout << "Warning: CLIP text model not found, text encoding disabled" << std::endl;
        textModelPath = "";
    }

    // 词表也是可选的（如果没有文本模型）
    std::string vocabPath = vocabPath_;
    if (!textModelPath.empty() && !fs::exists(vocabPath)) {
        std::cout << "Warning: Vocabulary file not found: " << vocabPath << std::endl;
        vocabPath = "";
    }

    // 创建CLIP编码器
    clipEncoder_ = std::make_unique<ClipEncoder>(
        visualModelPath,
        textModelPath,
        vocabPath,
        embeddingDim_
    );

    std::cout << "CLIP encoder initialized successfully!" << std::endl;
    std::cout << "  - Visual encoder: " << visualModelPath << std::endl;
    if (!textModelPath.empty()) {
        std::cout << "  - Text encoder: " << textModelPath << std::endl;
    }
    std::cout << "  - Embedding dimension: " << embeddingDim_ << std::endl;
}

void ModelManager::initializeCaptionModel() {
    std::string captionPath = (fs::path(modelPath_) / "blip_caption.onnx").string();
    if (!fs::exists(captionPath)) {
        std::cout << "Caption model not found, skipping: " << captionPath << std::endl;
        return;
    }
    captionModel_ = std::make_unique<CaptionModel>(env_, captionPath);
}

void ModelManager::initializeVqaModel() {
    std::string vqaPath = (fs::path(modelPath_) / "blip_vqa.onnx").string();
    if (!fs::exists(vqaPath)) {
        std::cout << "VQA model not found, skipping: " << vqaPath << std::endl;
        return;
    }
    // 选择词表：优先 BLIP 词表，其次 CLIP 词表
    std::string blipVocab = (fs::path(vocabPath_).parent_path() / "blip_vocab.txt").string();
    std::string vocabForVqa = fs::exists(blipVocab) ? blipVocab : vocabPath_;
    vqaModel_ = std::make_unique<VqaModel>(env_, vqaPath, vocabForVqa);
}

} // namespace core
} // namespace vindex
