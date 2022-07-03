from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from luxai2022.config import EnvConfig
from luxai2022.team import Team

from luxai2022.unit import Unit


@dataclass
class State:
    seed_rng: np.random.RandomState
    seed: int
    env_steps: int
    env_cfg: EnvConfig
    units: Dict[int, List[Unit]] = field(default_factory=dict)
    teams: Dict[str, Team] = field(default_factory=dict)
    