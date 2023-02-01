#include "agent.hpp"

#include "lux/action.hpp"
#include "lux/common.hpp"
#include "lux/log.hpp"

json Agent::setup() {
    if (step == 0) {
        return lux::BidAction(player == "player_1" ? "AlphaStrike" : "MotherMars", 10);
    }
    if (obs.teams[player].factories_to_place && isTurnToPlaceFactory()) {
        // transform spawn_mask to positions
        std::vector<lux::Position> spawns;
        const auto                &spawns_mask = obs.board.valid_spawns_mask;
        for (size_t x = 0; x < spawns_mask.size(); ++x) {
            for (size_t y = 0; y < spawns_mask[x].size(); ++y) {
                if (spawns_mask[x][y]) {
                    spawns.emplace_back(x, y);
                }
            }
        }
        static size_t index = 0;
        return lux::SpawnAction(spawns[(index += 7) % spawns.size()],
                                obs.teams[player].metal / 2,
                                obs.teams[player].water / 2);
    }
    return json::object();
}

json Agent::act() {
    json actions = json::object();
    for (const auto &[unitId, factory] : obs.factories[player]) {
        if (step % 4 < 3 && factory.canBuildLight(obs)) {
            actions[unitId] = factory.buildLight(obs);
        } else if (factory.canBuildHeavy(obs)) {
            actions[unitId] = factory.buildHeavy(obs);
        } else if (factory.canWater(obs)) {
            actions[unitId] = factory.water(obs);  // Alternatively set it to lux::FactoryAction::Water()
        }
    }
    for (const auto &[unitId, unit] : obs.units[player]) {
        for (int64_t i = 1; i < 5; ++i) {
            auto direction = lux::directionFromInt(i);
            auto moveCost = unit.moveCost(obs, direction);
            if (moveCost >= 0 && unit.power >= moveCost + unit.actionQueueCost(obs)) {
                LUX_LOG("ordering unit " << unit.unit_id << " to move in direction " << i);
                // Alternatively, push lux::UnitAction::Move(direction, 0)
                actions[unitId].push_back(unit.move(direction, 2));
                break;
            }
        }
    }
    // dump your created actions in a file by uncommenting this line
    // lux::dumpJsonToFile("last_actions.json", actions);
    // or log them by uncommenting this line
    // LUX_LOG(actions);
    return actions;
}
