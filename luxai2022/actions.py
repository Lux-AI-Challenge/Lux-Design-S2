from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np
from luxai2022.config import EnvConfig
from luxai2022.map.position import Position
from luxai2022.state import State

import luxai2022.unit as luxai_unit


# (0 = move, 1 = transfer X amount of R, 2 = pickup X amount of R, 3 = dig, 4 = self destruct, 5 = recharge X, 6 = repeat)
class Action:
    def __init__(self, act_type: str) -> None:
        self.act_type = act_type
        self.repeat = False

    def state_dict():
        raise NotImplementedError("")


class FactoryBuildAction(Action):
    def __init__(self, unit_type: int) -> None:
        super().__init__("factory_build")
        self.unit_type = unit_type

    def state_dict(self):
        return self.unit_type


class FactoryWaterAction(Action):
    def __init__(self) -> None:
        super().__init__("factory_water")

    def state_dict():
        return 2


class MoveAction(Action):
    def __init__(self, move_dir: int, dist: int = 1, repeat=False) -> None:
        super().__init__("move")
        # a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
        self.move_dir = move_dir
        self.dist = dist
        self.repeat = repeat

    def state_dict(self):
        return np.array([0, self.move_dir, self.dist, 0, self.repeat])


class TransferAction(Action):
    def __init__(self, transfer_dir: int, resource: int, transfer_amount: int, repeat=False) -> None:
        super().__init__("transfer")
        # a[2] = R = resource type (0 = ice, 1 = ore, 2 = water, 3 = metal, 4 power)
        self.transfer_dir = transfer_dir
        self.resource = resource
        self.transfer_amount = transfer_amount
        self.repeat = repeat

    def state_dict(self):
        return np.array([1, self.transfer_dir, self.resource, self.transfer_amount, self.repeat])


class PickupAction(Action):
    def __init__(self, resource: int, pickup_amount: int, repeat=False) -> None:
        super().__init__("pickup")
        # a[2] = R = resource type (0 = ice, 1 = ore, 2 = water, 3 = metal, 4 power)
        self.resource = resource
        self.pickup_amount = pickup_amount
        self.repeat = repeat

    def state_dict(self):
        return np.array([2, 0, self.resource, self.pickup_amount, self.repeat])


class DigAction(Action):
    def __init__(self, repeat=False) -> None:
        super().__init__("dig")
        self.repeat = repeat

    def state_dict(self):
        return np.array([3, 0, 0, 0, self.repeat])


class SelfDestructAction(Action):
    def __init__(self, repeat=False) -> None:
        super().__init__("self_destruct")
        self.repeat = repeat

    def state_dict(self):
        return np.array([4, 0, 0, 0, self.repeat])


class RechargeAction(Action):
    def __init__(self, power: int, repeat=False) -> None:
        super().__init__("recharge")
        self.power = power
        self.repeat = repeat

    def state_dict(self):
        return np.array([5, 0, 0, self.power, self.repeat])


def format_factory_action(a: int):
    if a == 0 or a == 1:
        return FactoryBuildAction(unit_type=a)
    elif a == 2:
        return FactoryWaterAction()
    else:
        raise ValueError(f"Action {a} for factory is invalid")


def format_action_vec(a: np.ndarray):
    # (0 = move, 1 = transfer X amount of R, 2 = pickup X amount of R, 3 = dig, 4 = self destruct, 5 = recharge X, 6 = repeat)
    a_type = a[0]
    if a_type == 0:
        return MoveAction(a[1], dist=1, repeat=a[4])
    elif a_type == 1:
        return TransferAction(a[1], a[2], a[3], repeat=a[4])
    elif a_type == 2:
        return PickupAction(a[2], a[3], repeat=a[4])
    elif a_type == 3:
        return DigAction(repeat=a[4])
    elif a_type == 4:
        return SelfDestructAction(repeat=a[4])
    elif a_type == 5:
        return RechargeAction(a[3], repeat=a[4])
    else:
        raise ValueError(f"Action {a} is invalid type, {a[0]} is not valid")


# a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])


def validate_actions(env_cfg: EnvConfig, state: State, actions_by_type):
    """
    validates actions and logs warnings for any invalid actions. Invalid actions are subsequently not evaluated
    """
    actions_by_type_validated = defaultdict(list)
    valid_action = True

    def invalidate_action(msg):
        nonlocal valid_action
        valid_action = False
        print(msg)

    for unit, transfer_action in actions_by_type["transfer"]:
        valid_action = True
        unit: luxai_unit.Unit
        transfer_action: TransferAction
        if transfer_action.resource > 4 or transfer_action.resource < 0:
            invalidate_action(
                f"Invalid Transfer Action for unit {unit}, transferring invalid resource id {transfer_action.resource}"
            )
        # if transfer_action.transfer_amount < 0:
        resource_id = transfer_action.resource
        amount = transfer_action.transfer_amount
        if resource_id == 0:
            if unit.cargo.ice < amount:
                invalidate_action(
                    f"Invalid Transfer Action for unit {unit} - Tried to transfer {amount} ice but only had {unit.cargo.ice}"
                )
        elif resource_id == 1:
            if unit.cargo.ore < amount:
                invalidate_action(
                    f"Invalid Transfer Action for unit {unit} - Tried to transfer {amount} ore but only had {unit.cargo.ore}"
                )
        elif resource_id == 2:
            if unit.cargo.water < amount:
                invalidate_action(
                    f"Invalid Transfer Action for unit {unit} - Tried to transfer {amount} water but only had {unit.cargo.water}"
                )
        elif resource_id == 3:
            if unit.cargo.metal < amount:
                invalidate_action(
                    f"Invalid Transfer Action for unit {unit} - Tried to transfer {amount} metal but only had {unit.cargo.metal}"
                )
        elif resource_id == 4:
            if unit.power < amount:
                invalidate_action(
                    f"Invalid Transfer Action for unit {unit} - Tried to transfer {amount} power but only had {unit.power}"
                )

        if valid_action:
            actions_by_type_validated["transfer"].append((unit, transfer_action))
    # TODO Resource Pickup
    for unit, pickup_action in actions_by_type["pickup"]:
        valid_action = True
        pickup_action: PickupAction
        unit: luxai_unit.Unit
        state.factories
        if valid_action:
            actions_by_type_validated["pickup"].append((unit, pickup_action))

    # TODO Movement
    for unit, move_action in actions_by_type["move"]:
        valid_action = True
        move_action: MoveAction
        target_pos: Position = unit.pos + move_action.dist * move_deltas[move_action.move_dir]
        if (
            target_pos.x < 0
            or target_pos.y < 0
            or target_pos.x >= state.board.width
            or target_pos.y >= state.board.height
        ):
            invalidate_action(
                f"Invalid movement action for unit {unit} - Tried to move to {target_pos} which is off the map"
            )
        rubble = state.board.rubble[target_pos.y, target_pos.x]
        power_required = unit.unit_cfg.MOVE_COST + unit.unit_cfg.RUBBLE_MOVEMENT_COST * rubble
        if power_required > unit.power:
            invalidate_action(
                f"Invalid movement action for unit {unit} - Tried to move to {target_pos} requiring {power_required} power but only had {unit.power}"
            )
        if valid_action:
            actions_by_type_validated["move"].append((unit, move_action))

    return actions_by_type_validated
