from typing import List
import numpy as np
from luxai2022.config import EnvConfig
from luxai2022.map.position import Position
from luxai2022.team import Team
from luxai2022.unit import UnitCargo


class Factory:
    def __init__(self, team: Team, unit_id: str) -> None:
        self.team_id = team.team_id
        self.team = team
        self.unit_id = unit_id
        self.pos = Position(np.zeros(2, dtype=int))
        self.power = 0
        self.cargo = UnitCargo()
        self.action_queue = [] # TODO can we queue actions or are factories outside of max control limit

    def refine_step(self, config: EnvConfig):
        consumed_ice = min(self.cargo.ice, config.FACTORY_PROCESSING_RATE_WATER)
        consumed_ore = min(self.cargo.metal, config.FACTORY_PROCESSING_RATE_METAL)

        self.cargo.ice -= consumed_ice
        self.cargo.ore -= consumed_ore
        
        # TODO - are we rounding or doing floats or anything?
        self.cargo.water += consumed_ice / config.ICE_WATER_RATIO
        self.cargo.metal += consumed_ore / config.ORE_METAL_RATIO

    def state_dict(self):
        return dict(
            pos=self.pos.pos,
            power=self.power,
            cargo=self.cargo.state_dict(),
            unit_id=self.unit_id,
            team_id=self.team_id

        )