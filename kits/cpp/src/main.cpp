#include <iostream>

#include "agent.hpp"
#include "lux/config.hpp"
#include "lux/json.hpp"
#include "lux/observation.hpp"

int main() {
    lux::EnvConfig config;
    while (std::cin && !std::cin.eof()) {
        json input;
        std::cin >> input;

        auto  step                 = input.at("step").get<int>();
        auto  playerString         = input.at("player").get<std::string>();
        auto  remainingOverageTime = input.at("remainingOverageTime").get<int>();
        Agent agent(step, playerString, remainingOverageTime);

        if (step == 0) {
            config = input.at("info").at("env_cfg").get<lux::EnvConfig>();
        }
        auto obs = input.at("obs").get<lux::Observation>();

        json output;
        // TODO determine value from board state
        if (step <= 3) {
            output = agent.setup(obs, config, input);
        } else {
            output = agent.act(obs, config, input);
        }
        std::cout << output << std::endl;
    }
    return 0;
}
