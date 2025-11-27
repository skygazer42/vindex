#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include "../index/faiss_index.h"
#include "../core/clip_encoder.h"

namespace vindex {
namespace index {

/**
 * @brief 简单的文本语料索引用于“图搜文”
 *
 * 读取文本列表文件（每行一条），用 CLIP 文本编码器生成向量，构建内存 FAISS 索引。
 * 仅用于演示/内置示例，不做持久化。
 */
class TextCorpusIndex {
public:
    struct Entry {
        int64_t id;
        std::string text;
    };

    explicit TextCorpusIndex(int dimension);

    /**
     * @brief 加载语料文件（每行一句文本），并编码构建索引
     * @param filePath 文件路径
     * @param encoder  CLIP 编码器（需要文本模型）
     * @return 是否成功
     */
    bool loadFromFile(const std::string& filePath, core::ClipEncoder& encoder);

    /**
     * @brief 搜索与图像特征最相似的文本
     */
    std::vector<std::pair<Entry, float>> search(const std::vector<float>& imageFeatures,
                                                int topK,
                                                float threshold = 0.0f) const;

    bool ready() const { return ready_; }
    size_t size() const { return entries_.size(); }

private:
    int dimension_;
    FaissIndex index_;
    std::unordered_map<int64_t, Entry> entryMap_;
    bool ready_;
};

} // namespace index
} // namespace vindex

