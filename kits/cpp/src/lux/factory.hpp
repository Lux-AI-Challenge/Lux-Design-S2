#pragma once

#include <cstdint>
#include <vector>

#include "lux/action.hpp"
#include "lux/common.hpp"
#include "lux/json.hpp"

namespace lux {
    struct Observation;
    struct Factory {
        Position                   pos;
        int64_t                    power;
        Cargo                      cargo;
        std::string                unit_id;
        int64_t                    strain_id;
        int64_t                    team_id;
        std::vector<FactoryAction> action_queue;

        int64_t buildHeavyMetalCost(const Observation &obs) const;
        int64_t buildHeavyPowerCost(const Observation &obs) const;
        bool    canBuildHeavy(const Observation &obs) const;
        int64_t buildLightMetalCost(const Observation &obs) const;
        int64_t buildLightPowerCost(const Observation &obs) const;
        bool    canBuildLight(const Observation &obs) const;
        int64_t waterCost(const Observation &obs) const;
        bool    canWater(const Observation &obs) const;

       private:
        int64_t buildMetalCost(const Observation &obs, const std::string &unitType) const;
        int64_t buildPowerCost(const Observation &obs, const std::string &unitType) const;
        bool    canBuild(const Observation &obs, const std::string &unitType) const;
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
