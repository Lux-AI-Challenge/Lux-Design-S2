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

        int64_t    moveCost(const Observation &obs, Direction direction) const;
        bool       canMove(const Observation &obs, Direction direction) const;
        UnitAction move(const Observation &obs, Direction direction, bool repeat = true) const;

        UnitAction transfer(const Observation &obs,
                            Direction          direction,
                            Resource           resource,
                            int64_t            amount,
                            bool               repeat = true) const;
        UnitAction pickup(const Observation &obs, Resource resource, int64_t amount, bool repeat = true) const;

        int64_t    digCost(const Observation &obs) const;
        bool       canDig(const Observation &obs) const;
        UnitAction dig(const Observation &obs, bool repeat = true) const;

        int64_t    selfDestructCost(const Observation &obs) const;
        bool       canSelfDestruct(const Observation &obs) const;
        UnitAction selfDestruct(const Observation &obs, bool repeat = true) const;

        UnitAction recharge(const Observation &obs, int64_t amount, bool repeat = true) const;
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Unit, team_id, unit_id, power, unit_type, pos, cargo)
}  // namespace lux
