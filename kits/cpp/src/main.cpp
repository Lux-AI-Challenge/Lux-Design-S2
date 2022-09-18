#include <iostream>

#include "agent.hpp"
#include "lux/config.hpp"
#include "lux/json.hpp"
#include "lux/log.hpp"
#include "lux/observation.hpp"

int main() {
    Agent agent;
    while (std::cin && !std::cin.eof()) {
        json input;
        std::cin >> input;

        lux::dumpJsonToFile("input.json", input);

        agent.step                 = input.at("step").get<int>();
        agent.player               = input.at("player").get<std::string>();
        agent.remainingOverageTime = input.at("remainingOverageTime").get<int>();

        auto obs = input.at("obs").get<lux::Observation>();
        if (agent.step == 0) {
            agent.config = input.at("info").at("env_cfg").get<lux::EnvConfig>();
            agent.obs    = obs;
        } else {
            // TODO figure out delta calculation
            agent.obs = obs;
        }

        json output;
        if (agent.step <= obs.board.factories_per_team + 1) {
            output = agent.setup();
        } else {
            agent.step = agent.obs.real_env_steps;
            output     = agent.act();
        }
        std::cout << output << std::endl;
    }
    return 0;
}
