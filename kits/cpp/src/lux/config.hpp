#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "lux/json.hpp"

namespace lux {
    struct UnitConfig {
        int64_t BATTERY_CAPACITY;
        int64_t CARGO_SPACE;
        int64_t CHARGE;
        int64_t DIG_COST;
        int64_t DIG_LICHEN_REMOVED;
        int64_t DIG_RESOURCE_GAIN;
        int64_t DIG_RUBBLE_REMOVED;
        int64_t INIT_POWER;
        int64_t METAL_COST;
        int64_t MOVE_COST;
        int64_t POWER_COST;
        int64_t RUBBLE_AFTER_DESTRUCTION;
        int64_t RUBBLE_MOVEMENT_COST;
        int64_t SELF_DESTRUCT_COST;
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

        const UnitConfig &operator[](const std::string &name) const;
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(UnitConfigs, HEAVY, LIGHT)

    struct WeatherConfig {
        std::map<std::string, int64_t> RUBBLE = {
            {"LIGHT", 0},
            {"HEAVY", 0}
        };
        double                 POWER_GAIN        = 1.0f;
        double                 POWER_CONSUMPTION = 1.0f;
        std::array<int64_t, 2> TIME_RANGE;
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(WeatherConfig, RUBBLE, POWER_GAIN, POWER_CONSUMPTION, TIME_RANGE)

    struct WeatherConfigs {
        WeatherConfig NONE;
        WeatherConfig COLD_SNAP;
        WeatherConfig DUST_STORM;
        WeatherConfig MARS_QUAKE;
        WeatherConfig SOLAR_FLARE;

        const WeatherConfig &operator[](const std::string &name) const;
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(WeatherConfigs, COLD_SNAP, DUST_STORM, MARS_QUAKE, SOLAR_FLARE)

    struct EnvConfig {
        bool                     BIDDING_SYSTEM;
        int64_t                  CYCLE_LENGTH;
        int64_t                  DAY_LENGTH;
        int64_t                  FACTORY_CHARGE;
        int64_t                  FACTORY_PROCESSING_RATE_METAL;
        int64_t                  FACTORY_PROCESSING_RATE_WATER;
        int64_t                  FACTORY_RUBBLE_AFTER_DESTRUCTION;
        int64_t                  FACTORY_WATER_CONSUMPTION;
        int64_t                  ICE_WATER_RATIO;
        int64_t                  INIT_POWER_PER_FACTORY;
        int64_t                  INIT_WATER_METAL_PER_FACTORY;
        int64_t                  LICHEN_GAINED_WITH_WATER;
        int64_t                  LICHEN_LOST_WITHOUT_WATER;
        int64_t                  LICHEN_WATERING_COST_FACTOR;
        int64_t                  MAX_FACTORIES;
        int64_t                  MAX_RUBBLE;
        int64_t                  MIN_FACTORIES;
        int64_t                  MIN_LICHEN_TO_SPREAD;
        std::vector<int64_t>     NUM_WEATHER_EVENTS_RANGE;
        int64_t                  ORE_METAL_RATIO;
        UnitConfigs              ROBOTS;
        int64_t                  UNITS_CONTROLLED;
        int64_t                  UNIT_ACTION_QUEUE_SIZE;
        WeatherConfigs           WEATHER;
        std::vector<std::string> WEATHER_ID_TO_NAME;
        int64_t                  map_size;
        int64_t                  max_episode_length;
        int64_t                  max_transfer_amount;
        bool                     validate_action_space;
        int64_t                  verbose;

        const WeatherConfig &getWeatherForId(int64_t id) const;
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
