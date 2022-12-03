#include "lux/observation.hpp"

namespace lux {

    const WeatherConfig &Observation::getCurrentWeather() const {
        return config.getWeatherForId(weather_schedule[real_env_steps]);
    }

    bool Observation::isDay() const { return real_env_steps % config.CYCLE_LENGTH < config.DAY_LENGTH; }

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
        if (o.board.factory_occupancy.size() != o.board.rubble.size()) {
            // init factory_occupancy map
            std::vector<int64_t> baseVec(o.board.rubble[0].size());
            o.board.factory_occupancy.resize(o.board.rubble.size(), baseVec);
        }
        // reset factory_occupancy map in case a factory died
        for (auto &row : o.board.factory_occupancy) {
            std::fill_n(row.begin(), row.size(), -1);
        }
        for (const auto &[_, factories] : o.factories) {
            for (const auto &[__, factory] : factories) {
                // TODO Is there a guarantee, that the factories are not placed on the edge??
                o.board.factory_occupancy[factory.pos.x - 1][factory.pos.y - 1] = factory.team_id;
                o.board.factory_occupancy[factory.pos.x - 1][factory.pos.y]     = factory.team_id;
                o.board.factory_occupancy[factory.pos.x - 1][factory.pos.y + 1] = factory.team_id;
                o.board.factory_occupancy[factory.pos.x][factory.pos.y - 1]     = factory.team_id;
                o.board.factory_occupancy[factory.pos.x][factory.pos.y]         = factory.team_id;
                o.board.factory_occupancy[factory.pos.x][factory.pos.y + 1]     = factory.team_id;
                o.board.factory_occupancy[factory.pos.x + 1][factory.pos.y - 1] = factory.team_id;
                o.board.factory_occupancy[factory.pos.x + 1][factory.pos.y]     = factory.team_id;
                o.board.factory_occupancy[factory.pos.x + 1][factory.pos.y + 1] = factory.team_id;
            }
        }
        // set unit configs
        for (auto &[_, units] : o.units) {
            for (auto &[__, unit] : units) {
                unit.unitConfig = o.config.ROBOTS[unit.unit_type];
            }
        }
    }
}  // namespace lux
