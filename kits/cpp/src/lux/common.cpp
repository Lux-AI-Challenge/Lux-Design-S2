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

    Position::Position(int64_t x_, int64_t y_) : x(x_), y(y_) {}

    Position Position::operator+(const Position &pos) const { return Position(x + pos.x, y + pos.y); }

    bool Position::operator==(const Position &pos) const { return x == pos.x && y == pos.y; }

    Position Position::Delta(Direction direction) {
        static std::array<Position, 5> deltas = {
            Position(0, 0),
            Position(0, -1),
            Position(1, 0),
            Position(0, 1),
            Position(-1, 0),
        };
        return deltas[std::underlying_type_t<Direction>(direction)];
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
