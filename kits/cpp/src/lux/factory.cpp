#include "lux/factory.hpp"

#include <numeric>

#include "lux/config.hpp"
#include "lux/observation.hpp"

namespace lux {
    int64_t Factory::buildMetalCost(const Observation &obs,

                                    const std::string &unitType) const {
        return obs.config.ROBOTS[unitType].METAL_COST;
    }

    int64_t Factory::buildPowerCost(const Observation &obs,
                                    const std::string &unitType) const {
        auto weather = obs.getCurrentWeather();
        return std::ceil(obs.config.ROBOTS[unitType].POWER_COST * weather.POWER_CONSUMPTION);
    }

    bool Factory::canBuild(const Observation &obs, const std::string &unitType) const {
        return power >= buildPowerCost(obs, unitType) && cargo.metal >= buildMetalCost(obs, unitType);
    }

    int64_t Factory::buildHeavyMetalCost(const Observation &obs) const { return buildMetalCost(obs, "HEAVY"); }

    int64_t Factory::buildHeavyPowerCost(const Observation &obs) const { return buildPowerCost(obs, "HEAVY"); }

    bool Factory::canBuildHeavy(const Observation &obs) const { return canBuild(obs, "HEAVY"); }

    FactoryAction Factory::buildHeavy(const Observation &obs) const {
        UNUSED(obs);
        LUX_ASSERT(canBuildHeavy(obs), "factory cannot build HEAVY, make sure to check beforehand with canBuildHeavy");
        return FactoryAction::BuildHeavy();
    }

    int64_t Factory::buildLightMetalCost(const Observation &obs) const { return buildMetalCost(obs, "LIGHT"); }

    int64_t Factory::buildLightPowerCost(const Observation &obs) const { return buildPowerCost(obs, "LIGHT"); }

    bool Factory::canBuildLight(const Observation &obs) const { return canBuild(obs, "LIGHT"); }

    FactoryAction Factory::buildLight(const Observation &obs) const {
        UNUSED(obs);
        LUX_ASSERT(canBuildLight(obs), "factory cannot build LIGHT, make sure to check beforehand with canBuildLight");
        return FactoryAction::BuildLight();
    }

    int64_t Factory::waterCost(const Observation &obs) const {
        auto countMatchingStrains = [&](int64_t sum, auto row) {
            return sum + std::count(row.begin(), row.end(), strain_id);
        };
        int64_t sum =
            std::accumulate(obs.board.lichen_strains.begin(), obs.board.lichen_strains.end(), 0, countMatchingStrains);
        return std::ceil(sum / obs.config.LICHEN_WATERING_COST_FACTOR) + 1;
    }

    bool Factory::canWater(const Observation &obs) const { return cargo.water >= waterCost(obs); }

    FactoryAction Factory::water(const Observation &obs) const {
        UNUSED(obs);
        LUX_ASSERT(canWater(obs), "factory cannot water, make sure to check beforehand with canWater");
        return FactoryAction::Water();
    }
}  // namespace lux
