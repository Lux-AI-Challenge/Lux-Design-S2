#include "lux/config.hpp"

#include "lux/exception.hpp"

namespace lux {
    const UnitConfig &UnitConfigs::operator[](const std::string &name) const {
        if (name == "HEAVY") {
            return HEAVY;
        }
        return LIGHT;
    }
}  // namespace lux
