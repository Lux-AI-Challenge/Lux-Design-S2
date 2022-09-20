#pragma once

#include <cstdint>
#include <vector>

#include "lux/action.hpp"
#include "lux/common.hpp"
#include "lux/json.hpp"

namespace lux {
    struct Unit {
        int64_t                 team_id;
        std::string             unit_id;
        int64_t                 power;
        std::string             unit_type;
        Position                pos;
        Cargo                   cargo;
        std::vector<UnitAction> action_queue;
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Unit, team_id, unit_id, power, unit_type, pos, cargo)
}  // namespace lux
