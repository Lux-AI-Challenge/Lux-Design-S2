from gym import spaces

from luxai2022.config import EnvConfig
from luxai2022.env import LuxAI2022


def get_act_space(self: LuxAI2022, config: EnvConfig, agent: int = 0):
    act_space = dict()

    for u in self.units[agent]:
        # TODO
        act_space[u.unit_id] = spaces.Box(0, 10, shape=(3,))

    return spaces.Dict(act_space)
