#pragma once

#include <array>
#include <cstdint>
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

    // TODO convenience functions to create actions

    struct UnitAction : public Action<std::array<int64_t, 5>> {
        enum class Type {
            MOVE,
            TRANSFER,
            PICKUP,
            DIG,
            SELF_DESTRUCT,
            RECHARGE,
        } type;
        enum class Direction {
            CENTER,
            UP,
            RIGHT,
            DOWN,
            LEFT,
        } direction;
        int64_t distance;
        enum class Resource {
            ICE,
            ORE,
            WATER,
            METAL,
            POWER,
        } resource;
        int64_t amount;
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

    struct FactoryAction : public Action<int64_t> {
        enum class Type {
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
