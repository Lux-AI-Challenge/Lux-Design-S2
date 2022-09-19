#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "lux/action.hpp"
#include "lux/json.hpp"

namespace lux {
    struct Cargo {
        int64_t ice;
        int64_t ore;
        int64_t water;
        int64_t metal;
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Cargo, ice, ore, water, metal)

    struct Unit {
        int64_t                team_id;
        std::string            unit_id;
        int64_t                power;
        std::string            unit_type;
        std::array<int64_t, 2> pos;
        Cargo                  cargo;
        std::vector<UnitAction> action_queue;
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Unit, team_id, unit_id, power, unit_type, pos, cargo)

    struct Team {
        int64_t              team_id;
        std::string          faction;
        int64_t              water;
        int64_t              metal;
        int64_t              factories_to_place;
        std::vector<int64_t> factory_strains;
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Team, team_id, faction, water, metal, factories_to_place, factory_strains)

    struct Factory {
        std::array<int64_t, 2>     pos;
        int64_t                    power;
        Cargo                      cargo;
        std::string                unit_id;
        int64_t                    strain_id;
        int64_t                    team_id;
        std::vector<FactoryAction> action_queue;
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(Factory,
                                                    pos,
                                                    power,
                                                    cargo,
                                                    unit_id,
                                                    strain_id,
                                                    team_id,
                                                    action_queue)

    struct Board {
        std::vector<std::vector<int64_t>>                          ice;
        std::vector<std::vector<int64_t>>                          lichen;
        std::vector<std::vector<int64_t>>                          lichen_strains;
        std::vector<std::vector<int64_t>>                          ore;
        std::vector<std::vector<int64_t>>                          rubble;
        std::map<std::string, std::vector<std::array<int64_t, 2>>> spawns;
        int64_t                                                    factories_per_team;

       private:
        bool                           initialized = false;
        std::map<std::string, int64_t> lichen_delta;
        std::map<std::string, int64_t> lichen_strains_delta;
        std::map<std::string, int64_t> rubble_delta;
        friend void from_json(const json &j, Board &b);
    };

    void to_json(json &j, const Board b);
    void from_json(const json &j, Board &b);

    struct Observation {
        Board                                                 board;
        std::map<std::string, std::map<std::string, Unit>>    units;
        std::map<std::string, Team>                           teams;
        std::map<std::string, std::map<std::string, Factory>> factories;
        std::vector<int64_t>                                  weather_schedule;
        int64_t                                               real_env_steps;

       private:
        bool initialized = false;
        friend void from_json(const json &j, Observation &o);
    };

    void to_json(json &j, const Observation o);
    void from_json(const json &j, Observation &o);
}  // namespace lux
