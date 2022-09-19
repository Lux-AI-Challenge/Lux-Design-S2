#include <iostream>

#include "agent.hpp"
#include "lux/json.hpp"
#include "lux/log.hpp"

int main() {
    Agent agent;
    while (std::cin && !std::cin.eof()) {
        json input;
        std::cin >> input;

        lux::dumpJsonToFile("input.json", input);

        input.at("step").get_to(agent.step);
        input.at("player").get_to(agent.player);
        input.at("remainingOverageTime").get_to(agent.remainingOverageTime);

        // parsing logic performs delta calculation
        input.at("obs").get_to(agent.obs);

        if (agent.step == 0) {
            input.at("info").at("env_cfg").get_to(agent.config);
        }

        json output;
        if (agent.step <= agent.obs.board.factories_per_team + 1) {
            output = agent.setup();
        } else {
            agent.step = agent.obs.real_env_steps;
            output     = agent.act();
        }
        std::cout << output << std::endl;
    }
    return 0;
}
