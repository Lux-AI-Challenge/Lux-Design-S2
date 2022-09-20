#include "lux/common.hpp"

#include <array>

#include "lux/exception.hpp"

namespace lux {

    Direction directionFromInt(int64_t raw) {
        LUX_ASSERT(raw >= 0 && raw <= 4, "got invalid UnitAction direction " + std::to_string(raw));
        return static_cast<Direction>(raw);
    }

    Resource resourceFromInt(int64_t raw) {
        LUX_ASSERT(raw >= 0 && raw <= 5, "got invalid UnitAction resource type " + std::to_string(raw));
        return static_cast<Resource>(raw);
    }

    void to_json(json &j, const Position a) {
        j.push_back(a.x);
        j.push_back(a.y);
    }

    void from_json(const json &j, Position &a) {
        a.x = j.at(0);
        a.y = j.at(1);
    }

}  // namespace lux
