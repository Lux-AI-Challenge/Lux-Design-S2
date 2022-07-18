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
                