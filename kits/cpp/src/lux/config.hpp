#pragma once

#include <array>
#include <string>
#include <vector>

#include "lux/json.hpp"

namespace lux {

    struct UnitConfig {
        int BATTERY_CAPACITY;
        int CARGO_SPACE;
        int CHARGE;
        int DIG_COST;
        int DIG_LICHEN_REMOVED;
        int DIG_RESOURCE_GAIN;
        int DIG_RUBBLE_REMOVED;
        int INIT_POWER;
        int METAL_COST;
        int MOVE_COST;
        int POWER_COST;
        int RUBBLE_AFTER_DESTRUCTION;
        int RUBBLE_MOVEMENT_COST;
        int SELF_DESTRUCT_COST;
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(UnitConfig,
                                       BATTERY_CAPACITY,
                                       CARGO_SPACE,
                                       CHARGE,
                                       DIG_COST,
                                       DIG_LICHEN_REMOVED,
                                       DIG_RESOURCE_GAIN,
                                       DIG_RUBBLE_REMOVED,
                                       INIT_POWER,
                                       METAL_COST,
                                       MOVE_COST,
                                       POWER_COST,
                                       RUBBLE_AFTER_DESTRUCTION,
                                       RUBBLE_MOVEMENT_COST,
                                       SELF_DESTRUCT_COST)

    struct UnitConfigs {
        UnitConfig HEAVY;
        UnitConfig LIGHT;

        UnitConfig &operator[](const std::string &name) {
            if (name == "HEAVY") {
                return HEAVY;
            }
            return LIGHT;
        }
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(UnitConfigs, HEAVY, LIGHT)

    struct WeatherConfig {
        // TODO: handle updated values for POWER_GAIN, RUBBLE and POWER_CONSUMPTION
        std::array<int, 2> TIME_RANGE;
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(WeatherConfig, TIME_RANGE)

    struct WeatherConfigs {
        WeatherConfig NONE;
        WeatherConfig COLD_SNAP;
        WeatherConfig DUST_STORM;
        WeatherConfig MARS_QUAKE;
        WeatherConfig SOLAR_FLARE;

        WeatherConfig &operator[](const std::string &name) {
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
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(WeatherConfigs, COLD_SNAP, DUST_STORM, MARS_QUAKE, SOLAR_FLARE)

    struct EnvConfig {
        bool                     BIDDING_SYSTEM;
        int                      CYCLE_LENGTH;
        int                      DAY_LENGTH;
        int                      FACTORY_CHARGE;
        int                      FACTORY_PROCESSING_RATE_METAL;
        int                      FACTORY_PROCESSING_RATE_WATER;
        int                      FACTORY_RUBBLE_AFTER_DESTRUCTION;
        int                      FACTORY_WATER_CONSUMPTION;
        int                      ICE_WATER_RATIO;
        int                      INIT_POWER_PER_FACTORY;
        int                      INIT_WATER_METAL_PER_FACTORY;
        int                      LICHEN_GAINED_WITH_WATER;
        int                      LICHEN_LOST_WITHOUT_WATER;
        int                      LICHEN_WATERING_COST_FACTOR;
        int                      MAX_FACTORIES;
        int                      MAX_RUBBLE;
        int                      MIN_FACTORIES;
        int                      MIN_LICHEN_TO_SPREAD;
        std::vector<int>         NUM_WEATHER_EVENTS_RANGE;
        int                      ORE_METAL_RATIO;
        UnitConfigs              ROBOTS;
        int                      UNITS_CONTROLLED;
        int                      UNIT_ACTION_QUEUE_SIZE;
        WeatherConfigs           WEATHER;
        std::vector<std::string> WEATHER_ID_TO_NAME;
        int                      map_size;
        int                      max_episode_length;
        int                      max_transfer_amount;
        bool                     validate_action_space;
        int                      verbose;
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(EnvConfig,
                                       BIDDING_SYSTEM,
                                       CYCLE_LENGTH,
                                       DAY_LENGTH,
                                       FACTORY_CHARGE,
                                       FACTORY_PROCESSING_RATE_METAL,
                                       FACTORY_PROCESSING_RATE_WATER,
                                       FACTORY_RUBBLE_AFTER_DESTRUCTION,
                                       FACTORY_WATER_CONSUMPTION,
                                       ICE_WATER_RATIO,
                                       INIT_POWER_PER_FACTORY,
                                       INIT_WATER_METAL_PER_FACTORY,
                                       LICHEN_GAINED_WITH_WATER,
                                       LICHEN_LOST_WITHOUT_WATER,
                                       LICHEN_WATERING_COST_FACTOR,
                                       MAX_FACTORIES,
                                       MAX_RUBBLE,
                                       MIN_FACTORIES,
                                       MIN_LICHEN_TO_SPREAD,
                                       NUM_WEATHER_EVENTS_RANGE,
                                       ORE_METAL_RATIO,
                                       ROBOTS,
                                       UNITS_CONTROLLED,
                                       UNIT_ACTION_QUEUE_SIZE,
                                       WEATHER,
                                       WEATHER_ID_TO_NAME,
                                       map_size,
                                       max_episode_length,
                                       max_transfer_amount,
                                       validate_action_space,
                                       verbose)

}  // namespace lux
