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
    return lux::SpawnAction(obs.board.spawns[player][index += 7], obs.teams[player].metal / 2, obs.teams[player].water);
}

json Agent::act() {
    json actions;
    for (auto [unitId, _] : obs.factories[player]) {
        if (step % 5 == 0) {
            actions[unitId] = lux::FactoryAction::BuildLight();
        }
    }
    for (auto [unitId, unit] : obs.units[player]) {
        actions[unitId].push_back(lux::UnitAction::Move(lux::UnitAction::Direction::UP, false));
    }
    return actions;
}
