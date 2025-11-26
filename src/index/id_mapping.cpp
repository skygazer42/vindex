#include "id_mapping.h"

namespace vindex {
namespace index {

std::string IdMapping::get(int64_t id) const {
    auto it = map_.find(id);
    if (it == map_.end()) {
        return {};
    }
    return it->second;
}

} // namespace index
} // namespace vindex
