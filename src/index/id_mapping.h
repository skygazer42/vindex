#pragma once

#include <string>
#include <unordered_map>

namespace vindex {
namespace index {

/**
 * @brief 简单的 ID ↔ 路径映射占位
 *
 * 未来可替换为 SQLite/更可靠的持久化实现。
 */
class IdMapping {
public:
    IdMapping() = default;
    ~IdMapping() = default;

    void set(int64_t id, const std::string& path) { map_[id] = path; }
    std::string get(int64_t id) const;
    bool contains(int64_t id) const { return map_.find(id) != map_.end(); }
    void clear() { map_.clear(); }

private:
    std::unordered_map<int64_t, std::string> map_;
};

} // namespace index
} // namespace vindex
