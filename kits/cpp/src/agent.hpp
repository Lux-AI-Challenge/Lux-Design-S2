#pragma once

#include <cstdint>
#include <vector>

#include "lux/config.hpp"
#include "lux/json.hpp"
#include "lux/observation.hpp"

struct Agent {
    int64_t          step;
    std::string      player;
    int64_t          remainingOverageTime;
    lux::Observation obs;

    Agent()                             = default;
    Agent(const Agent &other)           = delete;
    Agent(Agent &&other)                = delete;
    Agent operator=(const Agent &other) = delete;
    Agent operator=(Agent &&other)      = delete;
    ~Agent()                            = default;

    json setup();
    json act();
};
