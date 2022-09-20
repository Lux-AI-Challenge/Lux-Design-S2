#include "lux/factory.hpp"

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
        int64_t sum = 0;
        for (auto row : obs.board.lichen_strains) {
            for (auto strain : row) {
                if (strain == strain_id) {
                    ++sum;
                }
            }
        }
        return std::ceil(sum / config.LICHEN_WATERING_COST_FACTOR) + 1;
    }

    bool Factory::canWater(const Observation &obs, const EnvConfig &config) const {
        return cargo.water >= waterCost(obs, config);
    }
}  // namespace lux
