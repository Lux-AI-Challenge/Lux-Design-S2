#include "agent.hpp"

#include "lux/action.hpp"
#include "lux/common.hpp"
#include "lux/log.hpp"

json Agent::setup() {
    if (step == 0) {
        return lux::BidAction(player == "player_1" ? "AlphaStrike" : "MotherMars", 10);
    }
    static size_t index = 0;
    return lux::SpawnAction(obs.board.spawns[player][index += 7],
                            obs.teams[player].metal / 2,
                            obs.teams[player].water / 2);
}

json Agent::act() {
    json actions = json::object();
    for (const auto &[unitId, factory] : obs.factories[player]) {
        if (step % 4 < 3 && factory.canBuildLight(obs)) {
            actions[unitId] = factory.buildLight(obs);
        } else if (factory.canBuildHeavy(obs)) {
            actions[unitId] = factory.buildHeavy(obs);
        }
        if (factory.canWater(obs)) {
            actions[unitId] = factory.water(obs);  // Alternatively set it to lux::FactoryAction::Water()
        }
    }
    for (const auto &[unitId, unit] : obs.units[player]) {
        for (int64_t i = 0; i < 5; ++i) {
            auto direction = lux::directionFromInt(i);
            auto moveCost = unit.moveCost(obs, direction);
            if (moveCost >= 0 && unit.power >= moveCost + unit.actionQueueCost(obs)) {
                LUX_LOG("ordering unit " << unit.unit_id << " to move in direction " << i);
                // Alternatively, push lux::UnitAction::Move(direction, false)
                actions[unitId].push_back(unit.move(direction, false));
                break;
            }
        }
    }
    // dump your created actions in a file by uncommenting this line
    lux::dumpJsonToFile("last_actions.json", actions);
    return actions;
}
