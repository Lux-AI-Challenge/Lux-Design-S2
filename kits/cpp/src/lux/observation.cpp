#include "lux/observation.hpp"

namespace lux {
    void to_json(json &j, const Observation o) {
        j["board"]            = o.board;
        j["units"]            = o.units;
        j["teams"]            = o.teams;
        j["factories"]        = o.factories;
        j["weather_schedule"] = o.weather_schedule;
        j["real_env_steps"]   = o.real_env_steps;
    }
    void from_json(const json &j, Observation &o) {
        if (!o.initialized) {
            o.initialized = true;
            // weather only provided on step 0
            j.at("weather_schedule").get_to(o.weather_schedule);
        }
        j.at("board").get_to(o.board);
        j.at("units").get_to(o.units);
        j.at("teams").get_to(o.teams);
        j.at("factories").get_to(o.factories);
        j.at("real_env_steps").get_to(o.real_env_steps);
    }
}  // namespace lux
