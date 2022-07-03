from typing import List
from gym import spaces
import gym

from luxai2022.config import EnvConfig
from luxai2022.team import FactionTypes
from luxai2022.unit import Unit
import random

class FactionString(gym.Space):
    def __init__(
                self,
            ):
        self.valid_factions = ["None"] + [x.name for x in FactionTypes]

    def sample(self):
        return random.choice(self.valid_factions)

    def contains(self, x):
        return type(x) is "str" and x in self.valid_factions

def get_act_space(units: List[Unit], config: EnvConfig, agent: int = 0):
    act_space = dict()

    # for consistency, every action space per unit is fixed, makes it easier to work out of the box. 
    # we further annotate dimensions, in many places for clarity

    # for those who are programming rule-based bots, we provide helper functions to generate the action vector to store in an actions dict.

    # TODO - verify speed of building action spaces like this.
    for u in units[agent]:
        # Each action for any mobile unit, (light, heavy) is an array A of shape (max_queue, 4)
        # if max_queue is 1, we remove this dimension
        # let a be some element in A
        # Then
        
        # a[0] = action type 
        # (0 = move, 1 = transfer X amount of R, 2 = pickup X amount of R, 3 = dig, 4 = self destruct, 5 = recharge X, 6 = repeat)
        
        # a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
        
        # a[2] = R = resource type (0 = ice, 1 = ore, 2 = water, 3 = metal, 4 power)
        
        # a[3] = X, amount of resources transferred or picked up if action is transfer or pickup. 
        # If action is recharge, it is how much energy to store before executing the next action in queue

        act_space[u.unit_id] = spaces.MultiDiscrete([7, 5, 5, config.max_transfer_amount])

    # decide on a faction. We will only use what you select on the first turn
    act_space["faction"] = FactionString()

    # TODO add bidding

    return spaces.Dict(act_space)
