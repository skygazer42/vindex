#pragma once

#include <sqlite3.h>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include "faiss_index.h"

namespace vindex {

// 前向声明
namespace core {
class ClipEncoder;
}

namespace index {

/**
 * @brief 图像记录
 */
struct ImageRecord {
    int64_t id;               // 唯一ID
    std::string filePath;     // 图像文件路径
    std::string fileName;     // 文件名
    std::string category;     // 分类标签
    std::string description;  // 描述
    int64_t addTime;          // 添加时间戳
    int width;                // 图像宽度
    int height;               // 图像高度

    ImageRecord()
        : id(-1), addTime(0), width(0), height(0) {}
};

/**
 * @brief 图库数据库管理器
 *
 * 管理图像数据库和向量索引
 * - SQLite存储图像元数据
 * - FAISS存储图像特征向量
 * - 提供统一的增删改查接口
 */
class DatabaseManager {
public:
    /**
     * @brief 构造函数
     * @param dbPath SQLite数据库路径
     * @param indexPath FAISS索引路径
     * @param dimension 特征向量维度
     */
    explicit DatabaseManager(const std::string& dbPath,
                            const std::string& indexPath = "",
                            int dimension = 768);

    ~DatabaseManager();

    /**
     * @brief 初始化数据库（创建表结构）
     */
    bool initialize();

    /**
     * @brief 设置CLIP编码器（用于自动提取特征）
     */
    void setEncoder(core::ClipEncoder* encoder);

    // ==================== 图库管理 ====================

    /**
     * @brief 添加单张图像到库
     * @param imagePath 图像文件路径
     * @param category 分类标签（可选）
     * @param description 描述（可选）
     * @return 图像ID，失败返回-1
     */
    int64_t addImage(const std::string& imagePath,
                    const std::string& category = "",
                    const std::string& description = "");

    /**
     * @brief 批量添加图像
     * @param imagePaths 图像路径列表
     * @param category 分类标签
     * @return 成功添加的数量
     */
    size_t addImageBatch(const std::vector<std::string>& imagePaths,
                        const std::string& category = "");

    /**
     * @brief 从文件夹导入所有图像
     * @param folderPath 文件夹路径
     * @param recursive 是否递归子文件夹
     * @param progress 进度回调 (current, total)
     * @return 成功导入的数量
     */
    size_t importFolder(const std::string& folderPath,
                       bool recursive = false,
                       std::function<void(int, int)> progress = nullptr);

    /**
     * @brief 删除图像
     * @param id 图像ID
     * @return 是否删除成功
     */
    bool removeImage(int64_t id);

    /**
     * @brief 批量删除图像
     * @param ids 图像ID列表
     * @return 成功删除的数量
     */
    size_t removeImageBatch(const std::vector<int64_t>& ids);

    /**
     * @brief 更新图像信息
     * @param id 图像ID
     * @param category 新的分类（空字符串表示不更新）
     * @param description 新的描述（空字符串表示不更新）
     * @return 是否更新成功
     */
    bool updateImage(int64_t id,
                    const std::string& category = "",
                    const std::string& description = "");

    // ==================== 查询 ====================

    /**
     * @brief 根据ID查询图像记录
     */
    ImageRecord getById(int64_t id);

    /**
     * @brief 批量查询图像记录
     */
    std::vector<ImageRecord> getByIds(const std::vector<int64_t>& ids);

    /**
     * @brief 列出所有图像（分页）
     * @param offset 偏移量
     * @param limit 数量限制
     * @return 图像记录列表
     */
    std::vector<ImageRecord> listAll(int offset = 0, int limit = 100);

    /**
     * @brief 按分类查询
     */
    std::vector<ImageRecord> getByCategory(const std::string& category,
                                          int offset = 0,
                                          int limit = 100);

    /**
     * @brief 搜索文件名
     */
    std::vector<ImageRecord> searchByFileName(const std::string& keyword,
                                             int offset = 0,
                                             int limit = 100);

    /**
     * @brief 获取总数量
     */
    int64_t totalCount();

    /**
     * @brief 获取所有分类
     */
    std::vector<std::string> getAllCategories();

    // ==================== 向量搜索 ====================

    /**
     * @brief 图搜图
     * @param queryImagePath 查询图像路径
     * @param topK 返回Top-K个结果
     * @param threshold 相似度阈值
     * @return 搜索结果（图像记录 + 相似度分数）
     */
    struct SearchResultWithRecord {
        ImageRecord record;
        float score;

        SearchResultWithRecord(const ImageRecord& r, float s)
            : record(r), score(s) {}
    };

    std::vector<SearchResultWithRecord> searchByImage(
        const std::string& queryImagePath,
        int topK = 10,
        float threshold = 0.0f);

    /**
     * @brief 文搜图
     * @param queryText 查询文本
     * @param topK 返回Top-K个结果
     * @param threshold 相似度阈值
     * @return 搜索结果
     */
    std::vector<SearchResultWithRecord> searchByText(
        const std::string& queryText,
        int topK = 10,
        float threshold = 0.0f);

    // ==================== 索引管理 ====================

    /**
     * @brief 重建索引
     * @param progress 进度回调 (current, total)
     * @return 是否重建成功
     */
    bool rebuildIndex(std::function<void(int, int)> progress = nullptr);

    /**
     * @brief 保存索引到文件
     */
    bool saveIndex();

    /**
     * @brief 加载索引从文件
     */
    bool loadIndex();

    /**
     * @brief 获取FAISS索引引用
     */
    FaissIndex& faissIndex() { return faissIndex_; }

    /**
     * @brief 获取索引路径
     */
    std::string getIndexPath() const { return indexPath_; }

    /**
     * @brief 获取数据库文件路径
     */
    std::string getDbPath() const { return dbPath_; }

private:
    /**
     * @brief 执行SQL语句
     */
    bool executeSql(const std::string& sql);

    /**
     * @brief 提取图像特征
     */
    std::vector<float> extractFeatures(const std::string& imagePath);

    /**
     * @brief 获取图像尺寸
     */
    void getImageSize(const std::string& imagePath, int& width, int& height);

    /**
     * @brief 扫描文件夹中的图像文件
     */
    std::vector<std::string> scanImageFiles(const std::string& folderPath,
                                           bool recursive);

    /**
     * @brief 检查文件是否为支持的图像格式
     */
    bool isSupportedImageFormat(const std::string& filePath);

private:
    sqlite3* db_;                              // SQLite数据库连接
    FaissIndex faissIndex_;                    // FAISS向量索引
    std::string dbPath_;                       // 数据库文件路径
    std::string indexPath_;                    // 索引文件路径
    core::ClipEncoder* encoder_;               // CLIP编码器（不拥有）

    static const std::vector<std::string> supportedFormats_;
};

} // namespace index
} // namespace vindex
