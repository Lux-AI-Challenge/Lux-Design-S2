#pragma once

#include <cstdint>
#include <lux/json.hpp>

#define UNUSED(x) (void) x

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

    struct Cargo {
        int64_t ice;
        int64_t ore;
        int64_t water;
        int64_t metal;
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Cargo, ice, ore, water, metal)

    struct Position {
        int64_t x, y;

        Position() = default;
        Position(int64_t x_, int64_t y_);

        Position        operator+(const Position &pos) const;
        bool            operator==(const Position &pos) const;
        static Position Delta(Direction direction);
    };

    void to_json(json &j, const Position a);
    void from_json(const json &j, Position &a);
}  // namespace lux
