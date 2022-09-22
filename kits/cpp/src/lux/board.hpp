#pragma once

#include <map>
#include <string>
#include <vector>

#include "lux/common.hpp"
#include "lux/json.hpp"

namespace lux {
    struct Board {
        std::vector<std::vector<int64_t>>            ice;
        std::vector<std::vector<int64_t>>            lichen;
        std::vector<std::vector<int64_t>>            lichen_strains;
        std::vector<std::vector<int64_t>>            ore;
        std::vector<std::vector<int64_t>>            rubble;
        std::vector<std::vector<int64_t>>            factory_occupancy;  // populated in Observation deserialization
        std::map<std::string, std::vector<Position>> spawns;
        int64_t                                      factories_per_team;

       private:
        bool                           initialized = false;
        std::map<std::string, int64_t> lichen_delta;
        std::map<std::string, int64_t> lichen_strains_delta;
        std::map<std::string, int64_t> rubble_delta;
        friend void                    from_json(const json &j, Board &b);
    };

    void to_json(json &j, const Board b);
    void from_json(const json &j, Board &b);
}  // namespace lux
