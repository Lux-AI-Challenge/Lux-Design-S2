#pragma once

#include <vector>

#include "lux/config.hpp"
#include "lux/json.hpp"
#include "lux/observation.hpp"

class Agent {
    int         step;
    int         player;
    std::string playerString;
    int         remainingOverageTime;

   public:
    explicit Agent(int currentStep, std::string playerName, int currentRemainingOverageTime);
    Agent(const Agent &other) = delete;
    Agent(Agent &&other)      = delete;
    Agent operator=(const Agent &other) = delete;
    Agent operator=(Agent &&other)      = delete;
    ~Agent()                            = default;

    std::vector<json> setup(lux::Observation obs, lux::EnvConfig config, const json &raw);
    std::vector<json> act(lux::Observation obs, lux::EnvConfig config, const json &raw);
};
