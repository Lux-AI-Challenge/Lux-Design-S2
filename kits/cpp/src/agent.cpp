#include "agent.hpp"

Agent::Agent(int currentStep, std::string playerName, int currentRemainingOverageTime)
    : step(currentStep),
      player(playerName),
      remainingOverageTime(currentRemainingOverageTime) {}

std::vector<json> Agent::setup(lux::Observation obs, lux::EnvConfig config, const json &raw) {
    (void) obs;
    (void) config;
    (void) raw;
    return {};
}

std::vector<json> Agent::act(lux::Observation obs, lux::EnvConfig config, const json &raw) {
    (void) obs;
    (void) config;
    (void) raw;
    return {};
}
