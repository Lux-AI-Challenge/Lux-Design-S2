#include "lux/action.hpp"

#include <string>
#include <type_traits>

#include "lux/exception.hpp"

namespace lux {
    void populateUnitActionData(UnitAction &a) {
        a.data[0] = std::underlying_type_t<UnitAction::Type>(a.type);
        a.data[1] = std::underlying_type_t<UnitAction::Direction>(a.direction);
        a.data[3] = a.amount;
        a.data[4] = a.repeat ? 1 : 0;
    }

    void populateUnitActionMember(UnitAction &a) {
        if (a.data[0] > 5) {
            throw lux::Exception("got invalid UnitAction type " + std::to_string(a.data[0]));
        }
        if (a.data[1] > 4) {
            throw lux::Exception("got invalid UnitAction direction " + std::to_string(a.data[1]));
        }
        a.type      = static_cast<UnitAction::Type>(a.data[0]);
        a.direction = static_cast<UnitAction::Direction>(a.data[1]);
        a.amount    = a.data[3];
        a.repeat    = a.data[4] == 1;
    }

    void to_json(json &j, const UnitAction a) { j = a.data; }

    void from_json(const json &j, UnitAction &a) {
        j.get_to(a.data);
        populateUnitActionMember(a);
    }

    void to_json(json &j, const FactoryAction a) { j = a.data; }

    void from_json(const json &j, FactoryAction &a) {
        j.get_to(a.data);
        a.type = static_cast<FactoryAction::Type>(a.data);
        if (!a.isBuildAction() && !a.isWaterAction()) {
            throw lux::Exception("got invalid FactoryAction type " + std::to_string(a.data));
        }
    }
}  // namespace lux
