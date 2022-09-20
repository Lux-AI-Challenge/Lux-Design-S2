#pragma once

#include <cstdint>
#include <vector>

#include "lux/action.hpp"
#include "lux/common.hpp"
#include "lux/json.hpp"

namespace lux {
    struct Observation;
    struct EnvConfig;
    struct Unit {
        int64_t                 team_id;
        std::string             unit_id;
        int64_t                 power;
        std::string             unit_type;
        Position                pos;
        Cargo                   cargo;
        std::vector<UnitAction> action_queue;

        int64_t moveCost(const Observation &obs, const EnvConfig &config, Direction direction) const;
        bool    canMove(const Observation &obs, const EnvConfig &config, Direction direction) const;
        int64_t digCost(const Observation &obs, const EnvConfig &config) const;
        bool    canDig(const Observation &obs, const EnvConfig &config) const;
        int64_t selfDestructCost(const Observation &obs, const EnvConfig &config) const;
        bool    canSelfDestruct(const Observation &obs, const EnvConfig &config) const;
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Unit, team_id, unit_id, power, unit_type, pos, cargo)
}  // namespace lux
