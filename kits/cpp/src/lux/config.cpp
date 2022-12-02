#include "lux/config.hpp"

#include "lux/exception.hpp"

namespace lux {
    const UnitConfig &UnitConfigs::operator[](const std::string &name) const {
        if (name == "HEAVY") {
            return HEAVY;
        }
        return LIGHT;
    }

    const WeatherConfig &WeatherConfigs::operator[](const std::string &name) const {
        if (name == "COLD_SNAP") {
            return COLD_SNAP;
        }
        if (name == "DUST_STORM") {
            return DUST_STORM;
        }
        if (name == "MARS_QUAKE") {
            return MARS_QUAKE;
        }
        if (name == "SOLAR_FLARE") {
            return SOLAR_FLARE;
        }
        return NONE;
    }

    const WeatherConfig &EnvConfig::getWeatherForId(int64_t id) const {
        LUX_ASSERT(id >= 0 && static_cast<size_t>(id) < WEATHER_ID_TO_NAME.size(),
                   "invalid weather id for weather calculation " + std::to_string(id));
        return WEATHER[WEATHER_ID_TO_NAME[static_cast<size_t>(id)]];
    }
}  // namespace lux
