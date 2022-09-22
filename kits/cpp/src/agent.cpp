#include "agent.hpp"

#include "lux/action.hpp"
#include "lux/common.hpp"
#include "lux/log.hpp"

json Agent::setup() {
    if (step == 0) {
        return lux::BidAction(player == "player_1" ? "AlphaStrike" : "MotherMars", 10);
    }
    static size_t index = 0;
    return lux::SpawnAction(obs.board.spawns[player][index += 7], obs.teams[player].metal / 2, obs.teams[player].water);
}

json Agent::act() {
    json actions;
    for (auto [unitId, factory] : obs.factories[player]) {
        if (step % 4 < 3 && factory.canBuildLight(obs)) {
            actions[unitId] = factory.buildLight(obs);
        } else if (factory.canBuildHeavy(obs)) {
            actions[unitId] = factory.buildHeavy(obs);
        }
        if (factory.canWater(obs)) {
            actions[unitId] = factory.water(obs);
        }
    }
    for (auto [unitId, unit] : obs.units[player]) {
        for (int64_t i = 0; i < 5; ++i) {
            auto direction = lux::directionFromInt(i);
            if (unit.canMove(obs, direction)) {
                actions[unitId].push_back(unit.move(obs, direction, false));
                break;
            }
        }
    }
    return actions;
}
