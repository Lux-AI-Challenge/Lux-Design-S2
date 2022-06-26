from gym import spaces
from luxai2022.config import EnvConfig
from luxai2022.env import LuxAI2022
def get_act_space(self: LuxAI2022, config: EnvConfig, agent: int = 0):
    act_space = dict()

    for u in self.units[agent]:
        


    return spaces.Dict(act_space)
