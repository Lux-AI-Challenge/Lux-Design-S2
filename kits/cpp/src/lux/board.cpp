#include "lux/board.hpp"

#include "lux/exception.hpp"

namespace lux {
    namespace {
        void applyMappedDelta(std::vector<std::vector<int64_t>> &dest, const std::map<std::string, int64_t> &delta) {
            for (const auto &[k, v] : delta) {
                size_t offset = k.find_first_of(',');
                LUX_ASSERT(offset != k.npos, "mapping key not separated by comma");
                auto x     = static_cast<size_t>(std::stol(k.substr(0, offset)));
                auto y     = static_cast<size_t>(std::stol(k.substr(offset + 1)));
                dest[x][y] = v;
            }
        }
    }  // namespace

    void to_json(json &j, const Board b) {
        j["ice"]                = b.ice;
        j["lichen"]             = b.lichen;
        j["lichen_strains"]     = b.lichen_strains;
        j["ore"]                = b.ore;
        j["rubble"]             = b.rubble;
        j["valid_spawns_mask"]  = b.valid_spawns_mask;
        j["factories_per_team"] = b.factories_per_team;
    }

    void from_json(const json &j, Board &b) {
        if (!b.initialized) {
            b.initialized = true;
            // set initial board state
            j.at("ice").get_to(b.ice);
            j.at("lichen").get_to(b.lichen);
            j.at("lichen_strains").get_to(b.lichen_strains);
            j.at("ore").get_to(b.ore);
            j.at("rubble").get_to(b.rubble);
        } else {
            // apply delta for step > 0
            j.at("lichen").get_to(b.lichen_delta);
            j.at("lichen_strains").get_to(b.lichen_strains_delta);
            j.at("rubble").get_to(b.rubble_delta);
            applyMappedDelta(b.lichen, b.lichen_delta);
            applyMappedDelta(b.lichen_strains, b.lichen_strains_delta);
            applyMappedDelta(b.rubble, b.rubble_delta);
        }
        if (j.find("valid_spawns_mask") != j.end()) {
            j.at("valid_spawns_mask").get_to(b.valid_spawns_mask);
        }
        j.at("factories_per_team").get_to(b.factories_per_team);
    }
}  // namespace lux
