#include "faiss_index.h"
#include <algorithm>
#include <set>
#include <iostream>

namespace vindex {
namespace index {

FaissIndex::FaissIndex(int dimension, bool useGPU)
    : dimension_(dimension)
    , nextId_(0)
    , useGPU_(useGPU)
{
    // 创建基础L2索引（用于归一化向量，L2距离等价于余弦距离）
    baseIndex_ = std::make_unique<faiss::IndexFlatL2>(dimension);

    // 使用IDMap包装以支持自定义ID
    index_ = std::make_unique<faiss::IndexIDMap>(baseIndex_.get());
}

FaissIndex::~FaissIndex() {
    // IndexIDMap不拥有baseIndex_，需要手动管理
    // unique_ptr会自动释放
}

// ==================== 索引管理 ====================

bool FaissIndex::load(const std::string& indexPath) {
    try {
        // 读取索引文件
        faiss::Index* loadedIndex = faiss::read_index(indexPath.c_str());

        // 检查是否是IDMap类型
        faiss::IndexIDMap* idMapIndex = dynamic_cast<faiss::IndexIDMap*>(loadedIndex);
        if (!idMapIndex) {
            delete loadedIndex;
            std::cerr << "Error: Loaded index is not an IndexIDMap" << std::endl;
            return false;
        }

        // 检查维度是否匹配
        if (idMapIndex->d != dimension_) {
            delete loadedIndex;
            std::cerr << "Error: Index dimension mismatch. Expected: "
                     << dimension_ << ", Got: " << idMapIndex->d << std::endl;
            return false;
        }

        // IndexIDMap 默认拥有内部索引，避免 double-free：将所有权转移给 baseIndex_
        auto* flatIndex = dynamic_cast<faiss::IndexFlatL2*>(idMapIndex->index);
        if (!flatIndex) {
            delete loadedIndex;
            std::cerr << "Error: Loaded index base type is not IndexFlatL2" << std::endl;
            return false;
        }

        // 由 baseIndex_ 管理内部索引生命周期，IndexIDMap 不再删除它
        idMapIndex->own_fields = false;
        baseIndex_.reset(flatIndex);
        index_.reset(idMapIndex);

        // 更新nextId_（找到最大ID + 1）
        nextId_ = 0;
        idSet_.clear();
        if (index_->ntotal > 0) {
            // IndexIDMap 存储 id_map，直接读取以恢复 nextId_ 与集合
            for (faiss::idx_t id : idMapIndex->id_map) {
                idSet_.insert(static_cast<int64_t>(id));
                nextId_ = std::max(nextId_, static_cast<int64_t>(id) + 1);
            }
        }

        std::cout << "Loaded index with " << size() << " vectors" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Failed to load index: " << e.what() << std::endl;
        return false;
    }
}

bool FaissIndex::save(const std::string& indexPath) const {
    try {
        faiss::write_index(index_.get(), indexPath.c_str());
        std::cout << "Saved index with " << size() << " vectors to "
                 << indexPath << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to save index: " << e.what() << std::endl;
        return false;
    }
}

void FaissIndex::clear() {
    // 重新创建索引
    baseIndex_ = std::make_unique<faiss::IndexFlatL2>(dimension_);
    index_ = std::make_unique<faiss::IndexIDMap>(baseIndex_.get());
    nextId_ = 0;
    idSet_.clear();
}

// ==================== 向量操作 ====================

int64_t FaissIndex::add(const std::vector<float>& vector, int64_t id) {
    validateVector(vector);

    // 如果未指定ID，自动生成
    if (id < 0) {
        id = generateNewId();
    }

    // 添加到索引
    index_->add_with_ids(1, vector.data(), &id);
    idSet_.insert(id);

    return id;
}

void FaissIndex::addBatch(const std::vector<std::vector<float>>& vectors,
                         std::vector<int64_t>& ids,
                         const std::vector<int64_t>* inputIds) {
    if (vectors.empty()) {
        ids.clear();
        return;
    }

    // 验证所有向量
    for (const auto& vec : vectors) {
        validateVector(vec);
    }

    // 准备数据
    const size_t n = vectors.size();
    std::vector<float> flatVectors(n * dimension_);

    for (size_t i = 0; i < n; ++i) {
        std::copy(vectors[i].begin(), vectors[i].end(),
                 flatVectors.begin() + i * dimension_);
    }

    // 准备ID
    ids.resize(n);
    if (inputIds && inputIds->size() == n) {
        ids = *inputIds;
    } else {
        for (size_t i = 0; i < n; ++i) {
            ids[i] = generateNewId();
        }
    }

    // 批量添加
    index_->add_with_ids(n, flatVectors.data(), ids.data());
    idSet_.insert(ids.begin(), ids.end());
}

bool FaissIndex::remove(int64_t id) {
    try {
        faiss::IDSelectorBatch selector(1, &id);
        size_t removedCount = index_->remove_ids(selector);
        if (removedCount > 0) {
            idSet_.erase(id);
        }
        return removedCount > 0;
    } catch (const std::exception& e) {
        std::cerr << "Failed to remove ID " << id << ": " << e.what() << std::endl;
        return false;
    }
}

size_t FaissIndex::removeBatch(const std::vector<int64_t>& ids) {
    if (ids.empty()) {
        return 0;
    }

    try {
        faiss::IDSelectorBatch selector(ids.size(), ids.data());
        size_t removed = index_->remove_ids(selector);
        if (removed > 0) {
            for (int64_t id : ids) {
                idSet_.erase(id);
            }
        }
        return removed;
    } catch (const std::exception& e) {
        std::cerr << "Failed to remove IDs: " << e.what() << std::endl;
        return 0;
    }
}

// ==================== 搜索 ====================

std::vector<FaissIndex::SearchResult> FaissIndex::search(
    const std::vector<float>& queryVector,
    int topK,
    float threshold) const {

    validateVector(queryVector);

    if (empty()) {
        return {};
    }

    // 限制topK不超过索引大小
    topK = std::min(topK, static_cast<int>(size()));

    // 执行搜索
    std::vector<float> distances(topK);
    std::vector<int64_t> labels(topK);

    index_->search(1, queryVector.data(), topK,
                  distances.data(), labels.data());

    // 构建结果
    std::vector<SearchResult> results;
    results.reserve(topK);

    for (int i = 0; i < topK; ++i) {
        // 跳过无效ID（-1表示未找到）
        if (labels[i] < 0) {
            continue;
        }

        SearchResult result(labels[i], distances[i]);

        // 应用阈值过滤
        if (result.score >= threshold) {
            results.push_back(result);
        }
    }

    return results;
}

std::vector<std::vector<FaissIndex::SearchResult>> FaissIndex::searchBatch(
    const std::vector<std::vector<float>>& queryVectors,
    int topK,
    float threshold) const {

    if (queryVectors.empty() || empty()) {
        return {};
    }

    // 验证向量
    for (const auto& vec : queryVectors) {
        validateVector(vec);
    }

    const size_t nQueries = queryVectors.size();
    topK = std::min(topK, static_cast<int>(size()));

    // 准备查询数据
    std::vector<float> flatQueries(nQueries * dimension_);
    for (size_t i = 0; i < nQueries; ++i) {
        std::copy(queryVectors[i].begin(), queryVectors[i].end(),
                 flatQueries.begin() + i * dimension_);
    }

    // 执行批量搜索
    std::vector<float> distances(nQueries * topK);
    std::vector<int64_t> labels(nQueries * topK);

    index_->search(nQueries, flatQueries.data(), topK,
                  distances.data(), labels.data());

    // 构建结果
    std::vector<std::vector<SearchResult>> allResults(nQueries);

    for (size_t i = 0; i < nQueries; ++i) {
        std::vector<SearchResult> results;
        results.reserve(topK);

        for (int j = 0; j < topK; ++j) {
            int idx = i * topK + j;
            if (labels[idx] < 0) {
                continue;
            }

            SearchResult result(labels[idx], distances[idx]);
            if (result.score >= threshold) {
                results.push_back(result);
            }
        }

        allResults[i] = std::move(results);
    }

    return allResults;
}

// ==================== 信息获取 ====================

size_t FaissIndex::size() const {
    return index_->ntotal;
}

bool FaissIndex::contains(int64_t id) const {
    return idSet_.find(id) != idSet_.end();
}

// ==================== 私有方法 ====================

void FaissIndex::validateVector(const std::vector<float>& vector) const {
    if (vector.size() != static_cast<size_t>(dimension_)) {
        throw std::invalid_argument(
            "Vector dimension mismatch. Expected: " +
            std::to_string(dimension_) +
            ", Got: " + std::to_string(vector.size())
        );
    }
}

int64_t FaissIndex::generateNewId() {
    return nextId_++;
}

} // namespace index
} // namespace vindex
