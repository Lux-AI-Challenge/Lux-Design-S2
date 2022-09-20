#pragma once

#include <cstdint>
#include <vector>

#include "lux/action.hpp"
#include "lux/common.hpp"
#include "lux/json.hpp"

namespace lux {
    struct Factory {
        Position                   pos;
        int64_t                    power;
        Cargo                      cargo;
        std::string                unit_id;
        int64_t                    strain_id;
        int64_t                    team_id;
        std::vector<FactoryAction> action_queue;
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(Factory,
                                                    pos,
                                                    power,
                                                    cargo,
                                                    unit_id,
                                                    strain_id,
                                                    team_id,
                                                    action_queue)
}  // namespace lux
