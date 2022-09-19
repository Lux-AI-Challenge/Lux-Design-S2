#include "agent.hpp"

#include "lux/action.hpp"
#include "lux/log.hpp"

json Agent::setup() {
    if (step == 0) {
        std::string faction = "MotherMars";
        if (player == "player_1") {
            faction = "AlphaStrike";
        }
        return lux::BidAction(faction, 10);
    }
    static size_t index = 0;
    return lux::SpawnAction(obs.board.spawns[player][index++], obs.teams[player].metal / 2, obs.teams[player].water);
}

json Agent::act() {
    return {
        {"unit_0", {0, 0, 0, 0, 0}}
    };
}
