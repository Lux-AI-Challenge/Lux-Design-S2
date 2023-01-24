#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "lux/board.hpp"
#include "lux/config.hpp"
#include "lux/factory.hpp"
#include "lux/json.hpp"
#include "lux/team.hpp"
#include "lux/unit.hpp"

namespace lux {
    struct Observation {
        Board                                                 board;
        std::map<std::string, std::map<std::string, Unit>>    units;
        std::map<std::string, Team>                           teams;
        std::map<std::string, std::map<std::string, Factory>> factories;
        int64_t                                               real_env_steps;
        EnvConfig                                             config;  // is populated in main

        bool isDay() const;
    };

    void to_json(json &j, const Observation o);
    void from_json(const json &j, Observation &o);
}  // namespace lux
