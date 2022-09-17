#pragma once

#include "lux/json.hpp"

namespace lux {
    struct Observation {
        json obs;
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Observation, obs)
}  // namespace lux
