#include "lux/config.hpp"

namespace lux {
    UnitConfig &UnitConfigs::operator[](const std::string &name) {
        if (name == "HEAVY") {
            return HEAVY;
        }
        return LIGHT;
    }

    WeatherConfig &WeatherConfigs::operator[](const std::string &name) {
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
}  // namespace lux
