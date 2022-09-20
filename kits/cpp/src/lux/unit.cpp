#include "lux/unit.hpp"

#include <cmath>

#include "lux/config.hpp"
#include "lux/observation.hpp"

namespace lux {
    int64_t Unit::moveCost(const Observation &obs, const EnvConfig &config, Direction direction) const {
        Position target = pos + Position::Delta(direction);
        if (target.x < 0 || target.y < 0 || static_cast<size_t>(target.x) >= obs.board.rubble.size()
            || static_cast<size_t>(target.y) >= obs.board.rubble.size()) {
            return -1;
        }
        // TODO only detect opposing facories and improve performance
        bool occupiedByFactory = false;
        for (auto [player, factories] : obs.factories) {
            for (auto [_, factory] : factories) {
                if (factory.pos == target) {
                    occupiedByFactory = true;
                }
            }
        }
        if (occupiedByFactory) {
            return -1;
        }
        auto rubble  = obs.board.rubble[target.y][target.x];
        auto weather = config.getWeatherForId(obs.weather_schedule[obs.real_env_steps]);
        return std::ceil((config.ROBOTS[unit_type].MOVE_COST + config.ROBOTS[unit_type].RUBBLE_MOVEMENT_COST * rubble)
                         * weather.POWER_CONSUMPTION);
    }

    bool Unit::canMove(const Observation &obs, const EnvConfig &config, Direction direction) const {
        auto cost = moveCost(obs, config, direction);
        if (cost < 0) {
            return false;
        }
        return power >= cost;
    }

    int64_t Unit::digCost(const Observation &obs, const EnvConfig &config) const {
        auto weather = config.getWeatherForId(obs.weather_schedule[obs.real_env_steps]);
        return std::ceil(config.ROBOTS[unit_type].DIG_COST * weather.POWER_CONSUMPTION);
    }

    bool Unit::canDig(const Observation &obs, const EnvConfig &config) const { return power >= digCost(obs, config); }

    int64_t Unit::selfDestructCost(const Observation &obs, const EnvConfig &config) const {
        auto weather = config.getWeatherForId(obs.weather_schedule[obs.real_env_steps]);
        return std::ceil(config.ROBOTS[unit_type].SELF_DESTRUCT_COST * weather.POWER_CONSUMPTION);
    }

    bool Unit::canSelfDestruct(const Observation &obs, const EnvConfig &config) const {
        return power >= selfDestructCost(obs, config);
    }
}  // namespace lux
