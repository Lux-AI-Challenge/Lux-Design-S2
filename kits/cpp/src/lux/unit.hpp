#pragma once

#include <cstdint>
#include <vector>

#include "lux/action.hpp"
#include "lux/common.hpp"
#include "lux/config.hpp"
#include "lux/json.hpp"

namespace lux {
    struct Observation;
    struct Unit {
        int64_t                 team_id;
        std::string             unit_id;
        int64_t                 power;
        std::string             unit_type;
        Position                pos;
        Cargo                   cargo;
        std::vector<UnitAction> action_queue;
        UnitConfig              unitConfig;  // is populated in Observation deserialization

        int64_t actionQueueCost(const Observation &obs) const;

        int64_t    moveCost(const Observation &obs, Direction direction) const;
        UnitAction move(Direction direction, int64_t repeat = 0) const;

        UnitAction transfer(Direction direction, Resource resource, int64_t amount, int64_t repeat = 0) const;

        UnitAction pickup(Resource resource, int64_t amount, int64_t repeat = 0) const;

        int64_t    digCost(const Observation &obs) const;
        UnitAction dig(int64_t repeat = 0) const;

        int64_t    selfDestructCost(const Observation &obs) const;
        UnitAction selfDestruct(int64_t repeat = 0) const;

        UnitAction recharge(int64_t amount, int64_t repeat = 0) const;
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Unit, team_id, unit_id, power, unit_type, pos, cargo, action_queue)
}  // namespace lux
