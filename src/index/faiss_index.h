#pragma once

#include <faiss/IndexFlat.h>
#include <faiss/IndexIDMap.h>
#include <faiss/index_io.h>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>

namespace vindex {
namespace index {

/**
 * @brief FAISS向量索引封装
 *
 * 使用FAISS进行高效的相似度搜索
 * 支持：添加、删除、搜索、持久化
 */
class FaissIndex {
public:
    /**
     * @brief 搜索结果
     */
    struct SearchResult {
        int64_t id;           // 向量ID
        float distance;       // L2距离
        float score;          // 相似度分数 [0, 1]

        SearchResult(int64_t id_, float dist)
            : id(id_), distance(dist) {
            // 将L2距离转换为相似度分数
            // 对于归一化向量：L2距离 = sqrt(2 - 2*cosine_similarity)
            // 因此：cosine_similarity = (2 - L2^2) / 2
            float cosineSim = (2.0f - distance * distance) / 2.0f;
            score = std::max(0.0f, std::min(1.0f, cosineSim));
        }
    };

    /**
     * @brief 构造函数
     * @param dimension 向量维度（CLIP默认768）
     * @param useGPU 是否使用GPU加速（暂不支持，预留接口）
     */
    explicit FaissIndex(int dimension = 768, bool useGPU = false);
    ~FaissIndex();

    // ==================== 索引管理 ====================

    /**
     * @brief 从文件加载索引
     * @param indexPath 索引文件路径
     * @return 是否加载成功
     */
    bool load(const std::string& indexPath);

    /**
     * @brief 保存索引到文件
     * @param indexPath 索引文件路径
     * @return 是否保存成功
     */
    bool save(const std::string& indexPath) const;

    /**
     * @brief 清空索引
     */
    void clear();

    // ==================== 向量操作 ====================

    /**
     * @brief 添加单个向量
     * @param vector 特征向量
     * @param id 向量ID（如果为-1，自动分配）
     * @return 分配的ID
     */
    int64_t add(const std::vector<float>& vector, int64_t id = -1);

    /**
     * @brief 批量添加向量
     * @param vectors 特征向量列表
     * @param ids 输出：分配的ID列表
     * @param inputIds 输入：指定的ID列表（可选）
     */
    void addBatch(const std::vector<std::vector<float>>& vectors,
                  std::vector<int64_t>& ids,
                  const std::vector<int64_t>* inputIds = nullptr);

    /**
     * @brief 删除向量
     * @param id 向量ID
     * @return 是否删除成功
     */
    bool remove(int64_t id);

    /**
     * @brief 批量删除向量
     * @param ids 向量ID列表
     * @return 成功删除的数量
     */
    size_t removeBatch(const std::vector<int64_t>& ids);

    // ==================== 搜索 ====================

    /**
     * @brief 搜索最相似的向量
     * @param queryVector 查询向量
     * @param topK 返回Top-K个结果
     * @param threshold 相似度阈值（低于此值的结果会被过滤）
     * @return 搜索结果列表（按相似度降序）
     */
    std::vector<SearchResult> search(const std::vector<float>& queryVector,
                                    int topK = 10,
                                    float threshold = 0.0f) const;

    /**
     * @brief 批量搜索
     * @param queryVectors 查询向量列表
     * @param topK 每个查询返回Top-K个结果
     * @param threshold 相似度阈值
     * @return 每个查询的搜索结果
     */
    std::vector<std::vector<SearchResult>> searchBatch(
        const std::vector<std::vector<float>>& queryVectors,
        int topK = 10,
        float threshold = 0.0f) const;

    // ==================== 信息获取 ====================

    /**
     * @brief 获取索引中的向量数量
     */
    size_t size() const;

    /**
     * @brief 获取向量维度
     */
    int dimension() const { return dimension_; }

    /**
     * @brief 检查索引是否为空
     */
    bool empty() const { return size() == 0; }

    /**
     * @brief 检查ID是否存在
     */
    bool contains(int64_t id) const;

private:
    /**
     * @brief 验证向量维度
     */
    void validateVector(const std::vector<float>& vector) const;

    /**
     * @brief 生成新的ID
     */
    int64_t generateNewId();

private:
    int dimension_;                                // 向量维度
    std::unique_ptr<faiss::IndexFlatL2> baseIndex_;  // 基础L2索引
    std::unique_ptr<faiss::IndexIDMap> index_;     // 支持自定义ID的索引
    int64_t nextId_;                               // 下一个自动分配的ID
    bool useGPU_;                                  // 是否使用GPU
};

} // namespace index
} // namespace vindex
