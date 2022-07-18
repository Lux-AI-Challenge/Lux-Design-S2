from typing import Dict, List
from gym import spaces
import gym

from luxai2022.config import EnvConfig
from luxai2022.factory import Factory
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


def get_act_space_init(config: EnvConfig, agent: int = 0):
    # Get action space for turn 0 initialization
    # TODO add bidding
    act_space = dict()
    act_space["faction"] = FactionString()
    act_space["spawns"] = spaces.Box(low=0, high=config.map_size, shape=(config.MAX_FACTORIES, 2), dtype=int)
    return spaces.Dict(act_space)

def get_act_space(units: Dict[str, Dict[str, Unit]], factories: Dict[str, Dict[str, Factory]], config: EnvConfig, agent: int = 0):
    act_space = dict()

    # for consistency, every action space per unit is fixed, makes it easier to work out of the box. 
    # we further annotate dimensions, in many places for clarity

    # for those who are programming rule-based bots, we provide helper functions to generate the action vector to store in an actions dict, as well as generate
    # a human readable version of an action vector

    # TODO - verify speed of building action spaces like this.
    for u in units[agent].values():
        # Each action for any mobile unit, (light, heavy) is an array A of shape (max_queue, 5)
        # if max_queue is 1, we remove this dimension
        # let a be some element in A
        # Then
        
        # a[0] = action type 
        # (0 = move, 1 = transfer X amount of R, 2 = pickup X amount of R, 3 = dig, 4 = self destruct, 5 = recharge X, 6 = repeat)
        
        # a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
        
        # a[2] = R = resource type (0 = ice, 1 = ore, 2 = water, 3 = metal, 4 power)
        
        # a[3] = X, amount of resources transferred or picked up if action is transfer or pickup. 
        # If action is recharge, it is how much energy to store before executing the next action in queue

        # a[4] = 0,1 - repeat false or repeat true. If true, action is sent to end of queue once consumed

        act_space[u.unit_id] = spaces.MultiDiscrete([7, 5, 5, config.max_transfer_amount, 2])

    for factory in factories[agent].values():
        # action type (0 = build light robot, 1 = build heavy robot, 2 = water lichen)
        act_space[factory.unit_id] = spaces.Discrete(3)
    

    return spaces.Dict(act_space)
