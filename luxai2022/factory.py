from __future__ import annotations
from collections import deque
from typing import List
import numpy as np
from luxai2022.config import EnvConfig
from luxai2022.map.board import Board
from luxai2022.map.position import Position
from luxai2022.team import Team
from luxai2022.unit import UnitCargo
from luxai2022.actions import move_deltas

class Factory:
    def __init__(self, team: Team, unit_id: str, num_id: int) -> None:
        self.team_id = team.team_id
        self.team = team
        self.unit_id = unit_id
        self.pos = Position(np.zeros(2, dtype=int))
        self.power = 0
        self.cargo = UnitCargo()
        self.num_id = num_id
        self.action_queue = [] # TODO can we queue actions or are factories outside of max control limit

    def refine_step(self, config: EnvConfig):
        consumed_ice = min(self.cargo.ice, config.FACTORY_PROCESSING_RATE_WATER)
        consumed_ore = min(self.cargo.metal, config.FACTORY_PROCESSING_RATE_METAL)

        self.cargo.ice -= consumed_ice
        self.cargo.ore -= consumed_ore
        
        # TODO - are we rounding or doing floats or anything?
        self.cargo.water += consumed_ice / config.ICE_WATER_RATIO
        self.cargo.metal += consumed_ore / config.ORE_METAL_RATIO

    def water_cost(self, board: Board, env_cfg: EnvConfig):
        # TODO this can easily be a fairly slow function, can we make it much faster?
        
        # perform a BFS from the factory position and look for non rubble, non factory tiles.
        # find the current frontier from 4 starting positions x marked below
        """
             x
           _ _ _
          |     |
        x |     | x
          |_ _ _|
             x
        
        """
        seen = set()
        grow_lichen_positions = set()
        frontier = deque([
            self.pos.pos + np.array([0, -2]), self.pos.pos + np.array([2, 0]), self.pos.pos + np.array([0, 2]), self.pos.pos + np.array([-2, 0])
        ])
        H, W = board.lichen.shape
        while len(frontier) > 0:
            # consider pos for growing lichen and/or expanding
            pos = frontier.popleft()
            seen.add(pos)
            if pos[0] < 0 or pos[1] < 0 or pos[0] >= W or pos[1] >= H:
                # off map, ignore
                continue
            if board.rubble[pos[1], pos[0]] > 0:
                # has rubble, ignore
                continue
            if board.factory_occupancy_map[pos[1], pos[0]]:
                # factory is on this tile, ignore
                continue
            pos_lichen = board.lichen[pos[1], pos[0]]
            grow_lichen_positions.add(pos)
            
            if pos_lichen >= env_cfg.MIN_LICHEN_TO_SPREAD:
                for move_delta in move_deltas[1:]:
                    new_pos = pos + move_delta
                    if new_pos in seen: continue
                    frontier.append(new_pos)


    ### Add and sub resource functions copied over from unit.py code, can we consolidate them somewhere?
    def add_resource(self, resource_id, amount):
        if amount < 0: amount = 0
        if resource_id == 0:
            transfer_amount = min(self.cargo_space - self.cargo.ice, amount)
            self.cargo.ice += transfer_amount
        elif resource_id == 1:
            transfer_amount = min(self.cargo_space - self.cargo.ore, amount)
            self.cargo.ore += transfer_amount
        elif resource_id == 2:
            transfer_amount = min(self.cargo_space - self.cargo.water, amount)
            self.cargo.water += transfer_amount
        elif resource_id == 3:
            transfer_amount = min(self.cargo_space - self.cargo.metal, amount)
            self.cargo.metal += transfer_amount
        elif resource_id == 4:
            transfer_amount = min(self.battery_capacity - self.power, amount)
            self.power += transfer_amount
        return transfer_amount
    def sub_resource(self, resource_id, amount):
        # subtract/transfer out as much as you min(have, request)
        if amount < 0: amount = 0
        if resource_id == 0:
            transfer_amount = min(self.cargo.ice, amount)
            self.cargo.ice -= transfer_amount
        elif resource_id == 1:
            transfer_amount = min(self.cargo.ore, amount)
            self.cargo.ore -= transfer_amount
        elif resource_id == 2:
            transfer_amount = min(self.cargo.water, amount)
            self.cargo.water -= transfer_amount
        elif resource_id == 3:
            transfer_amount = min(self.cargo.metal, amount)
            self.cargo.metal -= transfer_amount
        elif resource_id == 4:
            transfer_amount = min(self.power, amount)
            self.power -= transfer_amount
        return transfer_amount




    def state_dict(self):
        return dict(
            pos=self.pos.pos,
            power=self.power,
            cargo=self.cargo.state_dict(),
            unit_id=self.unit_id,
            team_id=self.team_id

        )