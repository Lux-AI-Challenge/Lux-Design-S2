#include "lux/factory.hpp"

#include <numeric>

#include "lux/config.hpp"
#include "lux/observation.hpp"

namespace lux {
    int64_t Factory::buildMetalCost(const Observation & /*obs*/,
                                    const EnvConfig   &config,
                                    const std::string &unitType) const {
        return config.ROBOTS[unitType].METAL_COST;
    }

    int64_t Factory::buildPowerCost(const Observation &obs,
                                    const EnvConfig   &config,
                                    const std::string &unitType) const {
        auto weather = config.getWeatherForId(obs.weather_schedule[obs.real_env_steps]);
        return std::ceil(config.ROBOTS[unitType].POWER_COST * weather.POWER_CONSUMPTION);
    }

    bool Factory::canBuild(const Observation &obs, const EnvConfig &config, const std::string &unitType) const {
        return power >= buildPowerCost(obs, config, unitType) && cargo.metal >= buildMetalCost(obs, config, unitType);
    }

    int64_t Factory::buildHeavyMetalCost(const Observation &obs, const EnvConfig &config) const {
        return buildMetalCost(obs, config, "HEAVY");
    }

    int64_t Factory::buildHeavyPowerCost(const Observation &obs, const EnvConfig &config) const {
        return buildPowerCost(obs, config, "HEAVY");
    }

    bool Factory::canBuildHeavy(const Observation &obs, const EnvConfig &config) const {
        return canBuild(obs, config, "HEAVY");
    }

    int64_t Factory::buildLightMetalCost(const Observation &obs, const EnvConfig &config) const {
        return buildMetalCost(obs, config, "LIGHT");
    }

    int64_t Factory::buildLightPowerCost(const Observation &obs, const EnvConfig &config) const {
        return buildPowerCost(obs, config, "LIGHT");
    }

    bool Factory::canBuildLight(const Observation &obs, const EnvConfig &config) const {
        return canBuild(obs, config, "LIGHT");
    }

    int64_t Factory::waterCost(const Observation &obs, const EnvConfig &config) const {
        auto countMatchingStrains = [&](int64_t sum, auto row) {
            return sum + std::count(row.begin(), row.end(), strain_id);
        };
        int64_t sum =
            std::accumulate(obs.board.lichen_strains.begin(), obs.board.lichen_strains.end(), 0, countMatchingStrains);
        return std::ceil(sum / config.LICHEN_WATERING_COST_FACTOR) + 1;
    }

    bool Factory::canWater(const Observation &obs, const EnvConfig &config) const {
        return cargo.water >= waterCost(obs, config);
    }
}  // namespace lux
