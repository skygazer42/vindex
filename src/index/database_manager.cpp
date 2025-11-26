#include "database_manager.h"
#include "../core/clip_encoder.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <iostream>

namespace fs = std::filesystem;

namespace vindex {
namespace index {

const std::vector<std::string> DatabaseManager::supportedFormats_ = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"
};

DatabaseManager::DatabaseManager(const std::string& dbPath,
                                const std::string& indexPath,
                                int dimension)
    : db_(nullptr)
    , faissIndex_(dimension)
    , dbPath_(dbPath)
    , indexPath_(indexPath.empty() ? dbPath + ".index" : indexPath)
    , encoder_(nullptr)
{
}

DatabaseManager::~DatabaseManager() {
    if (db_) {
        sqlite3_close(db_);
    }
}

bool DatabaseManager::initialize() {
    // 打开SQLite数据库
    int rc = sqlite3_open(dbPath_.c_str(), &db_);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to open database: " << sqlite3_errmsg(db_) << std::endl;
        return false;
    }

    // 创建表结构
    const char* createTableSql = R"(
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY,
            file_path TEXT NOT NULL UNIQUE,
            file_name TEXT NOT NULL,
            category TEXT,
            description TEXT,
            add_time INTEGER NOT NULL,
            width INTEGER,
            height INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_category ON images(category);
        CREATE INDEX IF NOT EXISTS idx_file_name ON images(file_name);
        CREATE INDEX IF NOT EXISTS idx_add_time ON images(add_time);
    )";

    if (!executeSql(createTableSql)) {
        std::cerr << "Failed to create tables" << std::endl;
        return false;
    }

    // 尝试加载已有索引
    loadIndex();

    return true;
}

void DatabaseManager::setEncoder(core::ClipEncoder* encoder) {
    encoder_ = encoder;
}

// ==================== 图库管理 ====================

int64_t DatabaseManager::addImage(const std::string& imagePath,
                                  const std::string& category,
                                  const std::string& description) {
    if (!fs::exists(imagePath)) {
        std::cerr << "Image file does not exist: " << imagePath << std::endl;
        return -1;
    }

    if (!isSupportedImageFormat(imagePath)) {
        std::cerr << "Unsupported image format: " << imagePath << std::endl;
        return -1;
    }

    // 提取特征
    std::vector<float> features;
    try {
        features = extractFeatures(imagePath);
    } catch (const std::exception& e) {
        std::cerr << "Failed to extract features: " << e.what() << std::endl;
        return -1;
    }

    // 获取图像尺寸
    int width = 0, height = 0;
    getImageSize(imagePath, width, height);

    // 获取文件名
    std::string fileName = fs::path(imagePath).filename().string();

    // 获取当前时间戳
    auto now = std::chrono::system_clock::now();
    int64_t timestamp = std::chrono::duration_cast<std::chrono::seconds>(
        now.time_since_epoch()).count();

    // 插入数据库
    sqlite3_stmt* stmt;
    const char* sql = R"(
        INSERT INTO images (file_path, file_name, category, description, add_time, width, height)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    )";

    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare statement: " << sqlite3_errmsg(db_) << std::endl;
        return -1;
    }

    sqlite3_bind_text(stmt, 1, imagePath.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, fileName.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, category.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, description.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 5, timestamp);
    sqlite3_bind_int(stmt, 6, width);
    sqlite3_bind_int(stmt, 7, height);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE) {
        std::cerr << "Failed to insert record: " << sqlite3_errmsg(db_) << std::endl;
        return -1;
    }

    // 获取插入的ID
    int64_t imageId = sqlite3_last_insert_rowid(db_);

    // 添加到FAISS索引
    faissIndex_.add(features, imageId);

    return imageId;
}

size_t DatabaseManager::addImageBatch(const std::vector<std::string>& imagePaths,
                                     const std::string& category) {
    size_t successCount = 0;

    for (const auto& path : imagePaths) {
        if (addImage(path, category) >= 0) {
            successCount++;
        }
    }

    return successCount;
}

size_t DatabaseManager::importFolder(const std::string& folderPath,
                                    bool recursive,
                                    std::function<void(int, int)> progress) {
    auto imageFiles = scanImageFiles(folderPath, recursive);
    int total = imageFiles.size();
    int current = 0;
    size_t successCount = 0;

    for (const auto& imagePath : imageFiles) {
        if (addImage(imagePath) >= 0) {
            successCount++;
        }

        current++;
        if (progress) {
            progress(current, total);
        }
    }

    return successCount;
}

bool DatabaseManager::removeImage(int64_t id) {
    // 从数据库删除
    sqlite3_stmt* stmt;
    const char* sql = "DELETE FROM images WHERE id = ?";

    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return false;
    }

    sqlite3_bind_int64(stmt, 1, id);
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE) {
        return false;
    }

    // 从FAISS索引删除
    faissIndex_.remove(id);

    return true;
}

size_t DatabaseManager::removeImageBatch(const std::vector<int64_t>& ids) {
    size_t successCount = 0;

    for (int64_t id : ids) {
        if (removeImage(id)) {
            successCount++;
        }
    }

    return successCount;
}

bool DatabaseManager::updateImage(int64_t id,
                                 const std::string& category,
                                 const std::string& description) {
    std::string sql = "UPDATE images SET ";
    std::vector<std::string> updates;

    if (!category.empty()) {
        updates.push_back("category = ?");
    }
    if (!description.empty()) {
        updates.push_back("description = ?");
    }

    if (updates.empty()) {
        return true;  // 没有更新
    }

    sql += updates[0];
    for (size_t i = 1; i < updates.size(); ++i) {
        sql += ", " + updates[i];
    }
    sql += " WHERE id = ?";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return false;
    }

    int bindIndex = 1;
    if (!category.empty()) {
        sqlite3_bind_text(stmt, bindIndex++, category.c_str(), -1, SQLITE_TRANSIENT);
    }
    if (!description.empty()) {
        sqlite3_bind_text(stmt, bindIndex++, description.c_str(), -1, SQLITE_TRANSIENT);
    }
    sqlite3_bind_int64(stmt, bindIndex, id);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    return rc == SQLITE_DONE;
}

// ==================== 查询 ====================

ImageRecord DatabaseManager::getById(int64_t id) {
    ImageRecord record;

    sqlite3_stmt* stmt;
    const char* sql = "SELECT * FROM images WHERE id = ?";

    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return record;
    }

    sqlite3_bind_int64(stmt, 1, id);

    if (sqlite3_step(stmt) == SQLITE_ROW) {
        record.id = sqlite3_column_int64(stmt, 0);
        record.filePath = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        record.fileName = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));

        const char* category = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        record.category = category ? category : "";

        const char* desc = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        record.description = desc ? desc : "";

        record.addTime = sqlite3_column_int64(stmt, 5);
        record.width = sqlite3_column_int(stmt, 6);
        record.height = sqlite3_column_int(stmt, 7);
    }

    sqlite3_finalize(stmt);
    return record;
}

std::vector<ImageRecord> DatabaseManager::getByIds(const std::vector<int64_t>& ids) {
    std::vector<ImageRecord> records;
    records.reserve(ids.size());

    for (int64_t id : ids) {
        ImageRecord record = getById(id);
        if (record.id >= 0) {
            records.push_back(record);
        }
    }

    return records;
}

std::vector<ImageRecord> DatabaseManager::listAll(int offset, int limit) {
    std::vector<ImageRecord> records;

    sqlite3_stmt* stmt;
    const char* sql = "SELECT * FROM images ORDER BY add_time DESC LIMIT ? OFFSET ?";

    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return records;
    }

    sqlite3_bind_int(stmt, 1, limit);
    sqlite3_bind_int(stmt, 2, offset);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        ImageRecord record;
        record.id = sqlite3_column_int64(stmt, 0);
        record.filePath = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        record.fileName = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));

        const char* category = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        record.category = category ? category : "";

        const char* desc = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        record.description = desc ? desc : "";

        record.addTime = sqlite3_column_int64(stmt, 5);
        record.width = sqlite3_column_int(stmt, 6);
        record.height = sqlite3_column_int(stmt, 7);

        records.push_back(record);
    }

    sqlite3_finalize(stmt);
    return records;
}

std::vector<ImageRecord> DatabaseManager::getByCategory(const std::string& category,
                                                       int offset,
                                                       int limit) {
    std::vector<ImageRecord> records;

    sqlite3_stmt* stmt;
    const char* sql = "SELECT * FROM images WHERE category = ? ORDER BY add_time DESC LIMIT ? OFFSET ?";

    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return records;
    }

    sqlite3_bind_text(stmt, 1, category.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 2, limit);
    sqlite3_bind_int(stmt, 3, offset);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        ImageRecord record;
        record.id = sqlite3_column_int64(stmt, 0);
        record.filePath = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        record.fileName = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        record.category = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));

        const char* desc = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        record.description = desc ? desc : "";

        record.addTime = sqlite3_column_int64(stmt, 5);
        record.width = sqlite3_column_int(stmt, 6);
        record.height = sqlite3_column_int(stmt, 7);

        records.push_back(record);
    }

    sqlite3_finalize(stmt);
    return records;
}

std::vector<ImageRecord> DatabaseManager::searchByFileName(const std::string& keyword,
                                                          int offset,
                                                          int limit) {
    std::vector<ImageRecord> records;

    sqlite3_stmt* stmt;
    const char* sql = "SELECT * FROM images WHERE file_name LIKE ? ORDER BY add_time DESC LIMIT ? OFFSET ?";

    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return records;
    }

    std::string pattern = "%" + keyword + "%";
    sqlite3_bind_text(stmt, 1, pattern.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 2, limit);
    sqlite3_bind_int(stmt, 3, offset);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        ImageRecord record;
        record.id = sqlite3_column_int64(stmt, 0);
        record.filePath = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        record.fileName = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));

        const char* category = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        record.category = category ? category : "";

        const char* desc = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        record.description = desc ? desc : "";

        record.addTime = sqlite3_column_int64(stmt, 5);
        record.width = sqlite3_column_int(stmt, 6);
        record.height = sqlite3_column_int(stmt, 7);

        records.push_back(record);
    }

    sqlite3_finalize(stmt);
    return records;
}

int64_t DatabaseManager::totalCount() {
    sqlite3_stmt* stmt;
    const char* sql = "SELECT COUNT(*) FROM images";

    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return 0;
    }

    int64_t count = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        count = sqlite3_column_int64(stmt, 0);
    }

    sqlite3_finalize(stmt);
    return count;
}

std::vector<std::string> DatabaseManager::getAllCategories() {
    std::vector<std::string> categories;

    sqlite3_stmt* stmt;
    const char* sql = "SELECT DISTINCT category FROM images WHERE category IS NOT NULL AND category != ''";

    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return categories;
    }

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char* category = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        if (category) {
            categories.push_back(category);
        }
    }

    sqlite3_finalize(stmt);
    return categories;
}

// ==================== 向量搜索 ====================

std::vector<DatabaseManager::SearchResultWithRecord> DatabaseManager::searchByImage(
    const std::string& queryImagePath,
    int topK,
    float threshold) {

    // 提取查询图像特征
    std::vector<float> queryFeatures = extractFeatures(queryImagePath);

    // FAISS搜索
    auto searchResults = faissIndex_.search(queryFeatures, topK, threshold);

    // 获取图像记录
    std::vector<SearchResultWithRecord> results;
    results.reserve(searchResults.size());

    for (const auto& result : searchResults) {
        ImageRecord record = getById(result.id);
        if (record.id >= 0) {
            results.emplace_back(record, result.score);
        }
    }

    return results;
}

std::vector<DatabaseManager::SearchResultWithRecord> DatabaseManager::searchByText(
    const std::string& queryText,
    int topK,
    float threshold) {

    if (!encoder_) {
        throw std::runtime_error("Encoder not set");
    }

    // 编码文本
    std::vector<float> queryFeatures = encoder_->encodeText(queryText);

    // FAISS搜索
    auto searchResults = faissIndex_.search(queryFeatures, topK, threshold);

    // 获取图像记录
    std::vector<SearchResultWithRecord> results;
    results.reserve(searchResults.size());

    for (const auto& result : searchResults) {
        ImageRecord record = getById(result.id);
        if (record.id >= 0) {
            results.emplace_back(record, result.score);
        }
    }

    return results;
}

// ==================== 索引管理 ====================

bool DatabaseManager::rebuildIndex(std::function<void(int, int)> progress) {
    if (!encoder_) {
        std::cerr << "Encoder not set" << std::endl;
        return false;
    }

    // 清空现有索引
    faissIndex_.clear();

    // 获取所有图像记录
    auto allRecords = listAll(0, totalCount());
    int total = allRecords.size();
    int current = 0;

    for (const auto& record : allRecords) {
        try {
            // 提取特征
            std::vector<float> features = extractFeatures(record.filePath);

            // 添加到索引
            faissIndex_.add(features, record.id);

            current++;
            if (progress) {
                progress(current, total);
            }
        } catch (const std::exception& e) {
            std::cerr << "Failed to rebuild index for " << record.filePath
                     << ": " << e.what() << std::endl;
        }
    }

    // 保存索引
    return saveIndex();
}

bool DatabaseManager::saveIndex() {
    return faissIndex_.save(indexPath_);
}

bool DatabaseManager::loadIndex() {
    return faissIndex_.load(indexPath_);
}

// ==================== 私有方法 ====================

bool DatabaseManager::executeSql(const std::string& sql) {
    char* errMsg = nullptr;
    int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &errMsg);

    if (rc != SQLITE_OK) {
        std::cerr << "SQL error: " << errMsg << std::endl;
        sqlite3_free(errMsg);
        return false;
    }

    return true;
}

std::vector<float> DatabaseManager::extractFeatures(const std::string& imagePath) {
    if (!encoder_) {
        throw std::runtime_error("Encoder not set");
    }

    return encoder_->encodeImage(imagePath);
}

void DatabaseManager::getImageSize(const std::string& imagePath, int& width, int& height) {
    cv::Mat image = cv::imread(imagePath);
    if (!image.empty()) {
        width = image.cols;
        height = image.rows;
    }
}

std::vector<std::string> DatabaseManager::scanImageFiles(const std::string& folderPath,
                                                        bool recursive) {
    std::vector<std::string> imageFiles;

    if (!fs::exists(folderPath) || !fs::is_directory(folderPath)) {
        return imageFiles;
    }

    try {
        if (recursive) {
            for (const auto& entry : fs::recursive_directory_iterator(folderPath)) {
                if (entry.is_regular_file() && isSupportedImageFormat(entry.path().string())) {
                    imageFiles.push_back(entry.path().string());
                }
            }
        } else {
            for (const auto& entry : fs::directory_iterator(folderPath)) {
                if (entry.is_regular_file() && isSupportedImageFormat(entry.path().string())) {
                    imageFiles.push_back(entry.path().string());
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error scanning directory: " << e.what() << std::endl;
    }

    return imageFiles;
}

bool DatabaseManager::isSupportedImageFormat(const std::string& filePath) {
    std::string ext = fs::path(filePath).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    return std::find(supportedFormats_.begin(), supportedFormats_.end(), ext)
           != supportedFormats_.end();
}

} // namespace index
} // namespace vindex
