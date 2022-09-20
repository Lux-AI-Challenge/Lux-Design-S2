#pragma once

#include <cstdint>
#include <lux/json.hpp>

namespace lux {
    enum class Direction {
        CENTER,
        UP,
        RIGHT,
        DOWN,
        LEFT,
    };
    Direction directionFromInt(int64_t raw);

    enum class Resource {
        ICE,
        ORE,
        WATER,
        METAL,
        POWER,
    };
    Resource resourceFromInt(int64_t raw);

    struct Position {
        int64_t x, y;
    };

    void to_json(json &j, const Position a);
    void from_json(const json &j, Position &a);
}  // namespace lux
