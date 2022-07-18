from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

# (0 = move, 1 = transfer X amount of R, 2 = pickup X amount of R, 3 = dig, 4 = self destruct, 5 = recharge X, 6 = repeat)
class Action:
    def __init__(self, act_type: str) -> None:
        self.act_type = act_type
    def state_dict():
        raise NotImplementedError("")

class MoveAction(Action):
    def __init__(self, move_dir: int, dist: int = 1) -> None:
        super().__init__("move")
        # a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
        self.move_dir = move_dir
        self.dist = dist
    def state_dict(self):
        return np.array([0, self.move_dir, self.dist, 0])
class TransferAction(Action):
    def __init__(self, transfer_dir: int, resource: int, transfer_amount: int) -> None:
        super().__init__("transfer")
        # a[2] = R = resource type (0 = ice, 1 = ore, 2 = water, 3 = metal, 4 power)
        self.transfer_dir = transfer_dir
        self.resource = resource
        self.transfer_amount = transfer_amount
    def state_dict(self):
        return np.array([1, self.transfer_dir, self.resource, self.transfer_amount])

class PickupAction(Action):
    def __init__(self, resource: int, pickup_amount: int) -> None:
        super().__init__("pickup")
        # a[2] = R = resource type (0 = ice, 1 = ore, 2 = water, 3 = metal, 4 power)
        self.resource = resource
        self.pickup_amount = pickup_amount
    def state_dict(self):
        return np.array([2, 0, self.resource, self.pickup_amount])

class DigAction(Action):
    def __init__(self) -> None:
        super().__init__("dig")
    def state_dict(self):
        return np.array([3, 0, 0, 0])

class SelfDestructAction(Action):
    def __init__(self) -> None:
        super().__init__("self_destruct")
    def state_dict(self):
        return np.array([4, 0, 0, 0])

class RechargeAction(Action):
    def __init__(self, power: int) -> None:
        super().__init__("recharge")
        self.power = power
    def state_dict(self):
        return np.array([5, 0, 0, self.power])

def format_action_vec(a: np.ndarray):
    # (0 = move, 1 = transfer X amount of R, 2 = pickup X amount of R, 3 = dig, 4 = self destruct, 5 = recharge X, 6 = repeat)
    a_type = a[0]
    if a_type == 0:
        return MoveAction(a[1], dist=1)
    elif a_type == 1:
        return TransferAction(a[1], a[2], a[3])
    elif a_type == 2:
        return PickupAction(a[2], a[3])
    elif a_type == 3:
        return DigAction()
    elif a_type == 4:
        return SelfDestructAction()
    elif a_type == 5:
        return RechargeAction(a[3])
    else:
        raise ValueError(f"Action {a} is invalid type, {a[0]} is not valid")
                