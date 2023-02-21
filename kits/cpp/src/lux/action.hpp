#pragma once

#include <cstdint>
#include <string>

#include "lux/common.hpp"
#include "lux/exception.hpp"
#include "lux/json.hpp"

namespace lux {
    template<typename T>
    struct Action {
        using RawType = T;
        T raw;

        Action() = default;
        Action(T raw_) : raw(raw_) {}

        virtual void populateRaw()    = 0;
        virtual void populateMember() = 0;

        void toJson(json &j) {
            populateRaw();
            j = raw;
        }

        void fromJson(const json &j) {
            j.get_to(raw);
            populateMember();
        }
    };

    struct UnitAction final : public Action<std::array<int64_t, 6>> {
        enum class Type {
            MOVE,
            TRANSFER,
            PICKUP,
            DIG,
            SELF_DESTRUCT,
            RECHARGE,
        } type;
        Direction direction;
        Resource  resource;
        int64_t   amount;
        int64_t   repeat;
        int64_t   n;

        UnitAction() = default;
        UnitAction(UnitAction::RawType raw_);
        UnitAction(Type type_, Direction direction_, int64_t amount_, int64_t repeat_, int64_t n_);
        UnitAction(Type type_, Direction direction_, Resource resource_, int64_t amount_, int64_t repeat_, int64_t n_);

        static UnitAction Move(Direction direction, int64_t repeat, int64_t n);
        static UnitAction Transfer(Direction direction, Resource resource, int64_t amount, int64_t repeat, int64_t n);
        static UnitAction Pickup(Resource resource, int64_t amount, int64_t repeat, int64_t n);
        static UnitAction Dig(int64_t repeat, int64_t n);
        static UnitAction SelfDestruct(int64_t repeat, int64_t n);
        static UnitAction Recharge(int64_t amount, int64_t repeat, int64_t n);

        inline bool isMoveAction() const { return type == Type::MOVE; }
        inline bool isTransferAction() const { return type == Type::TRANSFER; }
        inline bool isPickupAction() const { return type == Type::PICKUP; }
        inline bool isDigAction() const { return type == Type::DIG; }
        inline bool isSelfDestructAction() const { return type == Type::SELF_DESTRUCT; }
        inline bool isRechargeAction() const { return type == Type::RECHARGE; }

       private:
        void populateRaw() override;
        void populateMember() override;
    };
    void to_json(json &j, const UnitAction a);
    void from_json(const json &j, UnitAction &a);

    struct FactoryAction final : public Action<int64_t> {
        enum class Type {
            BUILD_LIGHT,
            BUILD_HEAVY,
            WATER,
        } type;

        FactoryAction() = default;
        FactoryAction(FactoryAction::RawType raw_);
        FactoryAction(Type type_);

        static FactoryAction BuildLight();
        static FactoryAction BuildHeavy();
        static FactoryAction Water();

        inline bool isWaterAction() const { return type == Type::WATER; }
        inline bool isBuildAction() const { return type == Type::BUILD_HEAVY || type == Type::BUILD_LIGHT; }

        std::string getUnitType() const;

       private:
        void populateRaw() override;
        void populateMember() override;
    };
    void to_json(json &j, const FactoryAction a);
    void from_json(const json &j, FactoryAction &a);

    struct BidAction {
        std::string faction;
        int64_t     bid;

        BidAction() = default;
        BidAction(std::string faction_, int64_t bid_);
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(BidAction, faction, bid)

    struct SpawnAction {
        Position spawn;
        int64_t  metal;
        int64_t  water;

        SpawnAction() = default;
        SpawnAction(Position spawn_, int64_t metal_, int64_t water_);
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(SpawnAction, spawn, metal, water)
}  // namespace lux
