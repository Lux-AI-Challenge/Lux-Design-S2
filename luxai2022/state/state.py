from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from luxai2022.config import EnvConfig
from luxai2022.factory import Factory
from luxai2022.map.board import Board
from luxai2022.team import Team

from luxai2022.unit import Unit
from collections import OrderedDict
import copy

@dataclass
class State:
    seed_rng: np.random.RandomState
    seed: int
    env_steps: int
    env_cfg: EnvConfig
    board: Board = None
    units: Dict[str, Dict[str, Unit]] = field(default_factory=dict)
    factories: Dict[str, Dict[str, Factory]] = field(default_factory=dict)
    teams: Dict[str, Team] = field(default_factory=dict)
    global_id: int = 0
    
    def get_obs(self):
        units = dict()
        # TODO: speedups?
        for team in self.units:
            units[team] = dict()
            for unit in self.units[team].values():
                state_dict = unit.state_dict()
                if self.env_cfg.UNIT_ACTION_QUEUE_SIZE == 1:
                    # if config is such that action queue is size 1, we do not include the queue as it is always empty
                    del state_dict["action_queue"]
                units[team][unit.unit_id] = state_dict

        teams = dict()
        for k, v in self.teams.items():
            teams[k] = v.state_dict()
        
        factories = dict()
        for team in self.factories:
            factories[team] = dict()
            for factory in self.factories[team].values():
                state_dict = factory.state_dict()
                factories[team][factory.unit_id] = state_dict
        return copy.deepcopy(dict(
            units=units,
            team=teams,
            factories=factories,
            board=self.board.state_dict()
        ))
    def from_obs(obs):
        pass