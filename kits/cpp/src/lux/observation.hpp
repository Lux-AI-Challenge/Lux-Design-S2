#pragma once

#include <map>
#include <string>
#include <vector>

#include "lux/json.hpp"

namespace lux {
    struct Cargo {
        int ice;
        int ore;
        int water;
        int metal;
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Cargo, ice, ore, water, metal)

    struct Unit {
        int                team_id;
        std::string        unit_id;
        int                power;
        std::string        unit_type;
        std::array<int, 2> pos;
        Cargo              cargo;
        // TODO add action_queue
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Unit, team_id, unit_id, power, unit_type, pos, cargo)

    struct Team {
        int              team_id;
        std::string      faction;
        int              water;
        int              metal;
        int              factories_to_place;
        std::vector<int> factory_strains;
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Team, team_id, faction, water, metal, factories_to_place, factory_strains)

    struct Factory {
        std::array<int, 2> pos;
        int                power;
        Cargo              cargo;
        std::string        unit_id;
        int                strain_id;
        int                team_id;
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Factory, pos, power, cargo, unit_id, strain_id, team_id)

    struct Board {
        std::vector<std::vector<int>>                          ice;
        std::vector<std::vector<int>>                          lichen;
        std::vector<std::vector<int>>                          lichen_strains;
        std::vector<std::vector<int>>                          ore;
        std::vector<std::vector<int>>                          rubble;
        std::map<std::string, std::vector<std::array<int, 2>>> spawns;
        int                                                    factories_per_team;
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Board, ice, lichen, lichen_strains, ore, rubble, spawns, factories_per_team)

    struct Observation {
        Board                                                 board;
        std::map<std::string, std::map<std::string, Unit>>    units;
        std::map<std::string, Team>                           teams;
        std::map<std::string, std::map<std::string, Factory>> factories;
        std::vector<int>                                      weather_schedule;
        int                                                   real_env_steps;
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Observation, board, units, teams, factories, weather_schedule, real_env_steps)
}  // namespace lux
