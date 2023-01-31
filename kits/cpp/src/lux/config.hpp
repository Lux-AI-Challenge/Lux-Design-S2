#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "lux/json.hpp"

namespace lux {
    struct UnitConfig {
        int64_t ACTION_QUEUE_POWER_COST;
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
        double  RUBBLE_MOVEMENT_COST;
        int64_t SELF_DESTRUCT_COST;
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(UnitConfig,
                                       ACTION_QUEUE_POWER_COST,
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

    struct EnvConfig {
        bool        BIDDING_SYSTEM;
        int64_t     CYCLE_LENGTH;
        int64_t     DAY_LENGTH;
        int64_t     FACTORY_CHARGE;
        int64_t     FACTORY_PROCESSING_RATE_METAL;
        int64_t     FACTORY_PROCESSING_RATE_WATER;
        int64_t     FACTORY_RUBBLE_AFTER_DESTRUCTION;
        int64_t     FACTORY_WATER_CONSUMPTION;
        int64_t     ICE_WATER_RATIO;
        int64_t     INIT_POWER_PER_FACTORY;
        int64_t     INIT_WATER_METAL_PER_FACTORY;
        int64_t     LICHEN_GAINED_WITH_WATER;
        int64_t     LICHEN_LOST_WITHOUT_WATER;
        double      LICHEN_WATERING_COST_FACTOR;
        int64_t     MAX_LICHEN_PER_TILE;
        int64_t     MAX_FACTORIES;
        int64_t     MAX_RUBBLE;
        int64_t     MIN_FACTORIES;
        int64_t     MIN_LICHEN_TO_SPREAD;
        int64_t     ORE_METAL_RATIO;
        double      POWER_LOSS_FACTOR;
        int64_t     POWER_PER_CONNECTED_LICHEN_TILE;
        UnitConfigs ROBOTS;
        int64_t     UNIT_ACTION_QUEUE_SIZE;
        int64_t     map_size;
        int64_t     max_episode_length;
        int64_t     max_transfer_amount;
        bool        validate_action_space;
        int64_t     verbose;
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
                                       MAX_LICHEN_PER_TILE,
                                       MAX_FACTORIES,
                                       MAX_RUBBLE,
                                       MIN_FACTORIES,
                                       MIN_LICHEN_TO_SPREAD,
                                       ORE_METAL_RATIO,
                                       POWER_LOSS_FACTOR,
                                       POWER_PER_CONNECTED_LICHEN_TILE,
                                       ROBOTS,
                                       UNIT_ACTION_QUEUE_SIZE,
                                       map_size,
                                       max_episode_length,
                                       max_transfer_amount,
                                       validate_action_space,
                                       verbose)
}  // namespace lux
