#pragma once

#include <cstdint>
#include <vector>

#include "lux/json.hpp"

namespace lux {
    struct Team {
        int64_t              team_id;
        std::string          faction;
        int64_t              water;
        int64_t              metal;
        int64_t              factories_to_place;
        std::vector<int64_t> factory_strains;
        bool                 place_first;
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Team,
                                       team_id,
                                       faction,
                                       water,
                                       metal,
                                       factories_to_place,
                                       factory_strains,
                                       place_first)
}  // namespace lux
