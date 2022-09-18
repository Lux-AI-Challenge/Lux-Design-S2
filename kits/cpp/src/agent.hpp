#pragma once

#include <vector>

#include "lux/config.hpp"
#include "lux/json.hpp"
#include "lux/observation.hpp"

struct Agent {
    int              step;
    std::string      player;
    int              remainingOverageTime;
    lux::Observation obs;
    lux::EnvConfig   config;

    Agent()                             = default;
    Agent(const Agent &other)           = delete;
    Agent(Agent &&other)                = delete;
    Agent operator=(const Agent &other) = delete;
    Agent operator=(Agent &&other)      = delete;
    ~Agent()                            = default;

    json setup();
    json act();
};
