import sys
import numpy as np
from dataclasses import dataclass

from kits.python.lux.cargo import UnitCargo

# a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])

@dataclass
class Unit:
    team_id: int
    unit_id: str
    unit_type: str # "LIGHT" or "HEAVY"
    pos: np.ndarray
    power: int
    cargo: UnitCargo
    env_cfg: dict
    unit_cfg: dict
    # lichen tiles connected to this factory
    def move_cost(self, board, direction):
        target_pos = self.pos + move_deltas[direction]
        if target_pos[0] < 0 or target_pos[1] < 0 or target_pos[1] >= len(board) or target_pos[0] >= len(board[0]):
            print("Warning, tried to get move cost for going off the map", file=sys.stderr)
            return None
        rubble_at_target = board["rubble"][target_pos[1]][target_pos[0]]
        return self.unit_cfg.MOVE_COST + self.unit_cfg.RUBBLE_MOVEMENT_COST * rubble_at_target
    
    def move(self, direction, repeat=True):
        if isinstance(direction, int):
            direction = direction
        else:
            pass
        return np.array([0, direction, 0, 0, 1 if repeat else 0])

    def transfer(self, transfer_direction, transfer_resource, transfer_amount, repeat=True):
        assert transfer_resource < 5 and transfer_resource >= 0
        assert transfer_direction < 5 and transfer_direction >= 0
        return np.array([1, transfer_direction, transfer_resource, transfer_amount, 1 if repeat else 0])
    
    def pickup(self, pickup_resource, pickup_amount, repeat=True):
        assert pickup_resource < 5 and pickup_resource >= 0
        return np.array([2, 0, pickup_resource, pickup_amount, 1 if repeat else 0])
    
    def dig_cost(self):
        return self.unit_cfg.DIG_COST
    def can_dig(self):
        return self.power >= self.dig_cost()
    def dig(self, repeat=True):
        return np.array([3, 0, 0, 0, 1 if repeat else 0])

    def self_destruct_cost(self):
        return self.unit_cfg.SELF_DESTRUCT_COST
    def can_self_destruct(self):
        return self.power >= self.self_destruct_cost()
    def self_destruct(self, repeat=True):
        return np.array([4, 0, 0, 0, 1 if repeat else 0])

    def recharge(self, x, repeat=True):
        return np.array([5, 0, 0, x, 1 if repeat else 0])