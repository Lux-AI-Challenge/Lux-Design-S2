#pragma once

#include <array>
#include <string>

#include "lux/exception.hpp"
#include "lux/json.hpp"

namespace lux {
    template<typename T>
    struct Action {
        T data;
    };

    template<typename T>
    void to_json(json &j, const Action<T> a) {
        j = a.data;
    }

    template<typename T>
    void from_json(const json &j, Action<T> &a) {
        j.get_to(a.data);
    }

    struct UnitAction : public Action<std::array<unsigned int, 5>> {
        enum class Type : unsigned int {
            MOVE,
            TRANSFER,
            PICKUP,
            DIG,
            SELF_DESTRUCT,
            RECHARGE,
        } type;
        enum class Direction : unsigned int {
            CENTER,
            UP,
            RIGHT,
            DOWN,
            LEFT,
        } direction;
        // TODO figure out solution for distance/resource_type
        unsigned int amount;
        bool  repeat;

        inline bool isMoveAction() const { return type == Type::MOVE; }

        inline bool isTransferAction() const { return type == Type::TRANSFER; }

        inline bool isPickupAction() const { return type == Type::PICKUP; }

        inline bool isDigAction() const { return type == Type::DIG; }

        inline bool isSelfDestructAction() const { return type == Type::SELF_DESTRUCT; }

        inline bool isRechargeAction() const { return type == Type::RECHARGE; }
    };

    void to_json(json &j, const UnitAction a);
    void from_json(const json &j, UnitAction &a);

    struct FactoryAction : public Action<unsigned int> {
        enum class Type : unsigned int {
            BUILD_LIGHT,
            BUILD_HEAVY,
            WATER,
        } type;

        inline bool isWaterAction() const { return type == Type::WATER; }

        inline bool isBuildAction() const { return type == Type::BUILD_HEAVY || type == Type::BUILD_LIGHT; }

        std::string getUnitType() const {
            if (!isBuildAction()) {
                throw lux::Exception("cannot get build type from non-BuildAction");
            }
            return type == Type::BUILD_HEAVY ? "HEAVY" : "LIGHT";
        }
    };

    void to_json(json &j, const FactoryAction a);
    void from_json(const json &j, FactoryAction &a);

}  // namespace lux
