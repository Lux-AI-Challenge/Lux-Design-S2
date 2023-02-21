#include "lux/action.hpp"

#include <string>
#include <type_traits>

#include "lux/exception.hpp"

namespace lux {
    UnitAction::UnitAction(UnitAction::RawType raw_) : Action(raw_) { populateMember(); }

    UnitAction::UnitAction(Type type_, Direction direction_, int64_t amount_, int64_t repeat_, int64_t n_)
        : UnitAction(type_, direction_, resourceFromInt(0), amount_, repeat_, n_) {}

    UnitAction::UnitAction(Type      type_,
                           Direction direction_,
                           Resource  resource_,
                           int64_t   amount_,
                           int64_t   repeat_,
                           int64_t   n_)
        : type(type_),
          direction(direction_),
          resource(resource_),
          amount(amount_),
          repeat(repeat_),
          n(n_) {}

    UnitAction UnitAction::Move(Direction direction, int64_t repeat, int64_t n) {
        return UnitAction(Type::MOVE, direction, 0, repeat, n);
    }

    UnitAction UnitAction::Transfer(Direction direction, Resource resource, int64_t amount, int64_t repeat, int64_t n) {
        return UnitAction(Type::TRANSFER, direction, resource, amount, repeat, n);
    }

    UnitAction UnitAction::Pickup(Resource resource, int64_t amount, int64_t repeat, int64_t n) {
        return UnitAction(Type::PICKUP, Direction::CENTER, resource, amount, repeat, n);
    }

    UnitAction UnitAction::Dig(int64_t repeat, int64_t n) {
        return UnitAction(Type::DIG, Direction::CENTER, 0, repeat, n);
    }

    UnitAction UnitAction::SelfDestruct(int64_t repeat, int64_t n) {
        return UnitAction(Type::SELF_DESTRUCT, Direction::CENTER, 0, repeat, n);
    }

    UnitAction UnitAction::Recharge(int64_t amount, int64_t repeat, int64_t n) {
        return UnitAction(Type::RECHARGE, Direction::CENTER, amount, repeat, n);
    }

    void UnitAction::populateRaw() {
        raw[0] = std::underlying_type_t<UnitAction::Type>(type);
        raw[1] = std::underlying_type_t<Direction>(direction);
        raw[2] = std::underlying_type_t<Resource>(resource);
        raw[3] = amount;
        raw[4] = repeat;
        raw[5] = n;
    }

    void UnitAction::populateMember() {
        LUX_ASSERT(raw[0] >= 0 && raw[0] <= 5, "got invalid UnitAction type " + std::to_string(raw[0]));
        type      = static_cast<UnitAction::Type>(raw[0]);
        direction = directionFromInt(raw[1]);
        resource  = resourceFromInt(raw[2]);
        amount   = raw[3];
        repeat   = raw[4];
        n        = raw[5];
    }

    void to_json(json &j, const UnitAction a) {
        UnitAction copy = a;
        copy.toJson(j);
    }

    void from_json(const json &j, UnitAction &a) { a.fromJson(j); }

    FactoryAction::FactoryAction(FactoryAction::RawType raw_) : Action(raw_) { populateMember(); }

    FactoryAction::FactoryAction(Type type_) : Action(), type(type_) { populateRaw(); }

    FactoryAction FactoryAction::BuildLight() { return FactoryAction(Type::BUILD_LIGHT); }

    FactoryAction FactoryAction::BuildHeavy() { return FactoryAction(Type::BUILD_HEAVY); }

    FactoryAction FactoryAction::Water() { return FactoryAction(Type::WATER); }

    std::string FactoryAction::getUnitType() const {
        LUX_ASSERT(isBuildAction(), "cannot get build type from non-BuildAction");
        return type == Type::BUILD_HEAVY ? "HEAVY" : "LIGHT";
    }

    void FactoryAction::populateRaw() { raw = std::underlying_type_t<FactoryAction::Type>(type); }

    void FactoryAction::populateMember() {
        type = static_cast<FactoryAction::Type>(raw);
        LUX_ASSERT(isBuildAction() || isWaterAction(), "got invalid FactoryAction type " + std::to_string(raw));
    }

    void to_json(json &j, const FactoryAction a) {
        FactoryAction copy = a;
        copy.toJson(j);
    }

    void from_json(const json &j, FactoryAction &a) { a.fromJson(j); }

    BidAction::BidAction(std::string faction_, int64_t bid_) : faction(faction_), bid(bid_) {}

    SpawnAction::SpawnAction(Position spawn_, int64_t metal_, int64_t water_)
        : spawn(spawn_),
          metal(metal_),
          water(water_) {}
}  // namespace lux
