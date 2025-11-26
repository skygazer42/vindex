/**
 * 文本编码测试程序
 * 用于验证 CLIP 文本编码器是否正常工作
 */

#include "core/model_manager.h"
#include "core/clip_encoder.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace vindex::core;

void printVector(const std::vector<float>& vec, int maxElements = 10) {
    std::cout << "[";
    for (size_t i = 0; i < std::min(vec.size(), static_cast<size_t>(maxElements)); ++i) {
        std::cout << std::fixed << std::setprecision(4) << vec[i];
        if (i < std::min(vec.size(), static_cast<size_t>(maxElements)) - 1) {
            std::cout << ", ";
        }
    }
    if (vec.size() > maxElements) {
        std::cout << " ... (" << vec.size() - maxElements << " more)";
    }
    std::cout << "]" << std::endl;
}

float vectorNorm(const std::vector<float>& vec) {
    float sum = 0.0f;
    for (float v : vec) {
        sum += v * v;
    }
    return std::sqrt(sum);
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "CLIP 文本编码测试程序" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    try {
        // 1. 配置模型路径
        std::cout << "步骤 1: 配置模型路径..." << std::endl;

        ModelManager& modelManager = ModelManager::instance();
        modelManager.setModelPath("./assets/models");
        modelManager.setVocabPath("./assets/vocab/bpe_simple_vocab_16e6.txt");
        modelManager.setEmbeddingDim(768);

        std::cout << "  ✓ 模型目录: ./assets/models" << std::endl;
        std::cout << "  ✓ 词表路径: ./assets/vocab/bpe_simple_vocab_16e6.txt" << std::endl;
        std::cout << std::endl;

        // 2. 加载 CLIP 编码器
        std::cout << "步骤 2: 加载 CLIP 编码器..." << std::endl;

        ClipEncoder& encoder = modelManager.clipEncoder();

        if (!encoder.hasTextEncoder()) {
            std::cerr << "  ✗ 文本编码器未加载！" << std::endl;
            std::cerr << "  请确保以下文件存在：" << std::endl;
            std::cerr << "    - assets/models/clip_text.onnx" << std::endl;
            std::cerr << "    - assets/vocab/bpe_simple_vocab_16e6.txt" << std::endl;
            return 1;
        }

        std::cout << "  ✓ CLIP 编码器加载成功" << std::endl;
        std::cout << "  ✓ 特征维度: " << encoder.getEmbeddingDim() << std::endl;
        std::cout << std::endl;

        // 3. 测试用例
        std::vector<std::string> testTexts = {
            "a cat",
            "a dog sitting on grass",
            "sunset over the ocean",
            "red sports car",
            "person wearing glasses"
        };

        if (argc > 1) {
            // 使用命令行参数作为测试文本
            testTexts.clear();
            for (int i = 1; i < argc; ++i) {
                testTexts.push_back(argv[i]);
            }
        }

        std::cout << "步骤 3: 测试文本编码..." << std::endl;
        std::cout << std::endl;

        for (size_t i = 0; i < testTexts.size(); ++i) {
            const auto& text = testTexts[i];

            std::cout << "测试 " << (i + 1) << ": \"" << text << "\"" << std::endl;
            std::cout << std::string(50, '-') << std::endl;

            try {
                // 编码文本
                auto features = encoder.encodeText(text);

                // 验证特征向量
                std::cout << "  特征维度: " << features.size() << std::endl;

                // 计算向量模长
                float norm = vectorNorm(features);
                std::cout << "  向量模长: " << std::fixed << std::setprecision(6)
                         << norm << std::endl;

                // 显示前10个值
                std::cout << "  前10个值: ";
                printVector(features, 10);

                // 统计信息
                float minVal = *std::min_element(features.begin(), features.end());
                float maxVal = *std::max_element(features.begin(), features.end());
                float avgVal = 0.0f;
                for (float v : features) avgVal += v;
                avgVal /= features.size();

                std::cout << "  统计信息:" << std::endl;
                std::cout << "    最小值: " << minVal << std::endl;
                std::cout << "    最大值: " << maxVal << std::endl;
                std::cout << "    平均值: " << avgVal << std::endl;

                // 验证归一化
                if (std::abs(norm - 1.0f) < 0.01f) {
                    std::cout << "  ✓ 向量已正确归一化" << std::endl;
                } else {
                    std::cout << "  ⚠ 警告：向量可能未正确归一化 (期望模长=1.0)" << std::endl;
                }

            } catch (const std::exception& e) {
                std::cout << "  ✗ 编码失败: " << e.what() << std::endl;
            }

            std::cout << std::endl;
        }

        // 4. 批量编码测试
        std::cout << "步骤 4: 测试批量编码..." << std::endl;
        std::cout << std::string(50, '-') << std::endl;

        try {
            auto batchFeatures = encoder.encodeTextBatch(testTexts);

            std::cout << "  批量大小: " << testTexts.size() << std::endl;
            std::cout << "  输出数量: " << batchFeatures.size() << std::endl;

            for (size_t i = 0; i < batchFeatures.size(); ++i) {
                float norm = vectorNorm(batchFeatures[i]);
                std::cout << "  文本 " << (i + 1) << " 模长: " << norm << std::endl;
            }

            std::cout << "  ✓ 批量编码成功" << std::endl;

        } catch (const std::exception& e) {
            std::cout << "  ✗ 批量编码失败: " << e.what() << std::endl;
        }

        std::cout << std::endl;

        // 5. 相似度测试
        std::cout << "步骤 5: 测试文本间相似度..." << std::endl;
        std::cout << std::string(50, '-') << std::endl;

        std::vector<std::pair<std::string, std::string>> similarityTests = {
            {"a cat", "a dog"},
            {"a cat", "a feline animal"},
            {"sunset", "sunrise"},
            {"car", "automobile"}
        };

        for (const auto& [text1, text2] : similarityTests) {
            auto features1 = encoder.encodeText(text1);
            auto features2 = encoder.encodeText(text2);

            float similarity = ClipEncoder::cosineSimilarity(features1, features2);

            std::cout << "  \"" << text1 << "\" vs \"" << text2 << "\"" << std::endl;
            std::cout << "    相似度: " << std::fixed << std::setprecision(4)
                     << similarity << std::endl;
        }

        std::cout << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "测试完成！" << std::endl;
        std::cout << "========================================" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << std::endl;
        std::cerr << "✗ 错误: " << e.what() << std::endl;
        std::cerr << std::endl;
        std::cerr << "请确保：" << std::endl;
        std::cerr << "1. ONNX 模型文件已导出到 assets/models/" << std::endl;
        std::cerr << "2. BPE 词表文件存在于 assets/vocab/" << std::endl;
        std::cerr << "3. ONNX Runtime 库已正确安装" << std::endl;
        return 1;
    }
}
