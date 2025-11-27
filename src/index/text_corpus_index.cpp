#include "text_corpus_index.h"
#include <fstream>
#include <sstream>
#include <algorithm>

namespace vindex {
namespace index {

TextCorpusIndex::TextCorpusIndex(int dimension)
    : dimension_(dimension)
    , index_(dimension)
    , ready_(false) {
}

bool TextCorpusIndex::loadFromFile(const std::string& filePath, core::ClipEncoder& encoder) {
    std::ifstream fin(filePath);
    if (!fin.is_open()) {
        return false;
    }

    std::vector<std::string> texts;
    std::string line;
    while (std::getline(fin, line)) {
        // 去掉行尾空白
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
            line.pop_back();
        }
        if (line.empty()) continue;
        texts.push_back(line);
    }

    if (texts.empty()) {
        return false;
    }

    // 编码文本（需要文本模型）
    std::vector<std::vector<float>> features = encoder.encodeTextBatch(texts);
    if (features.empty() || features[0].size() != static_cast<size_t>(dimension_)) {
        return false;
    }

    // 构建索引
    std::vector<int64_t> ids;
    ids.reserve(texts.size());
    index_.clear();
    entryMap_.clear();

    for (size_t i = 0; i < texts.size(); ++i) {
        int64_t id = static_cast<int64_t>(i);
        ids.push_back(id);
        entryMap_.emplace(id, Entry{id, texts[i]});
    }
    index_.addBatch(features, ids, nullptr);

    ready_ = true;
    return true;
}

std::vector<std::pair<TextCorpusIndex::Entry, float>> TextCorpusIndex::search(
    const std::vector<float>& imageFeatures,
    int topK,
    float threshold) const {
    if (!ready_) {
        return {};
    }

    auto results = index_.search(imageFeatures, topK, threshold);
    std::vector<std::pair<Entry, float>> out;
    out.reserve(results.size());
    for (const auto& r : results) {
        auto it = entryMap_.find(r.id);
        if (it != entryMap_.end()) {
            out.emplace_back(it->second, r.score);
        }
    }
    return out;
}

} // namespace index
} // namespace vindex

