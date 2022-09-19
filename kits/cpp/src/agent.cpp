#include "agent.hpp"

#include "lux/log.hpp"

json Agent::setup() {
    if (step == 0) {
        std::string faction = "MotherMars";
        if (player == "player_1") {
            faction = "AlphaStrike";
        }
        return {
            {"faction", faction},
            {    "bid",      10}
        };
    }
    return {
        {"spawn", obs.board.spawns[player][0]},
        {"metal",                           2},
        {"water",                           2},
    };
}

json Agent::act() {
    return {
        {"unit_0", {0, 0, 0, 0, 0}}
    };
}
