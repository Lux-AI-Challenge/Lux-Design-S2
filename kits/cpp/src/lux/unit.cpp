#include "lux/unit.hpp"

#include <cmath>

#include "lux/config.hpp"
#include "lux/exception.hpp"
#include "lux/observation.hpp"

namespace lux {
    int64_t Unit::moveCost(const Observation &obs, Direction direction) const {
        Position target = pos + Position::Delta(direction);
        if (target.x < 0 || target.y < 0 || static_cast<size_t>(target.x) >= obs.board.rubble.size()
            || static_cast<size_t>(target.y) >= obs.board.rubble.size()) {
            return -1;
        }
        auto factoryTeam = obs.board.factory_occupancy[target.y][target.x];
        if (factoryTeam != -1 && team_id != factoryTeam) {
            return -1;
        }
        auto rubble  = obs.board.rubble[target.y][target.x];
        auto weather = obs.getCurrentWeather();
        return std::ceil((unitConfig.MOVE_COST + unitConfig.RUBBLE_MOVEMENT_COST * rubble) * weather.POWER_CONSUMPTION);
    }

    bool Unit::canMove(const Observation &obs, Direction direction) const {
        auto cost = moveCost(obs, direction);
        if (cost < 0) {
            return false;
        }
        return power >= cost;
    }

    UnitAction Unit::move(const Observation &obs, Direction direction, bool repeat) const {
        LUX_ASSERT(canMove(obs, direction), "unit cannot move there, make sure to check beforehand with canMove");
        return UnitAction::Move(direction, repeat);
    }

    UnitAction Unit::transfer(const Observation & /*obs*/,
                              Direction direction,
                              Resource  resource,
                              int64_t   amount,
                              bool      repeat) const {
        return UnitAction::Transfer(direction, resource, amount, repeat);
    }

    UnitAction Unit::pickup(const Observation & /*obs*/, Resource resource, int64_t amount, bool repeat) const {
        return UnitAction::Pickup(resource, amount, repeat);
    }

    int64_t Unit::digCost(const Observation &obs) const {
        auto weather = obs.getCurrentWeather();
        return std::ceil(unitConfig.DIG_COST * weather.POWER_CONSUMPTION);
    }

    bool Unit::canDig(const Observation &obs) const { return power >= digCost(obs); }

    UnitAction Unit::dig(const Observation &obs, bool repeat) const {
        LUX_ASSERT(canDig(obs), "unit cannot dig here, make sure to check beforehand with canDig");
        return UnitAction::Dig(repeat);
    }

    int64_t Unit::selfDestructCost(const Observation &obs) const {
        auto weather = obs.getCurrentWeather();
        return std::ceil(unitConfig.SELF_DESTRUCT_COST * weather.POWER_CONSUMPTION);
    }

    bool Unit::canSelfDestruct(const Observation &obs) const { return power >= selfDestructCost(obs); }

    UnitAction Unit::selfDestruct(const Observation &obs, bool repeat) const {
        LUX_ASSERT(canSelfDestruct(obs),
                   "unit cannot self destruct, make sure to check beforehand with canSelfDestruct");
        return UnitAction::SelfDestruct(repeat);
    }

    UnitAction Unit::recharge(const Observation & /*obs*/, int64_t amount, bool repeat) const {
        return UnitAction::Recharge(amount, repeat);
    }
}  // namespace lux
