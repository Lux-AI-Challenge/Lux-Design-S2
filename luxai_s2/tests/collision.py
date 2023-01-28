import time

import numpy as np

from luxai_s2.env import LuxAI_S2
from luxai_s2.utils import my_turn_to_place_factory
from luxai_s2.utils.heuristics.factory_placement import place_near_random_ice
from luxai_s2.unit import UnitType
from luxai_s2.actions import MoveAction
import math
# TODO: replace with pytest some day
def test_heavy_entering_only():
    env: LuxAI_S2 = LuxAI_S2()
    obs = env.reset(seed=0)
    np.random.seed(0)
    while env.state.real_env_steps < 0:
        if env.env_steps == 0:
            actions = dict()
            for p in env.agents:
                actions[p] = dict(bid=0, faction="AlphaStrike")
            obs, _, _, _ = env.step(actions)
        else:
            actions = dict()
            for p in env.agents:
                if my_turn_to_place_factory(env.state.teams[p].place_first, env.state.env_steps):
                    action = place_near_random_ice(p, obs[p])
                else:
                    action = dict()
                actions[p] = action
            obs, _, _, _ = env.step(actions)
        # env.render()
        # time.sleep(0.5)
    
    # add heavy
    u1 = env.add_unit(env.state.teams["player_0"], UnitType.HEAVY, np.array([20, 10]))
    u1.power = 10
    u1.action_queue = [MoveAction(1)]
    u2 = env.add_unit(env.state.teams["player_0"], UnitType.HEAVY, np.array([20, 10]))
    u2.power = 5
    u2.action_queue = [MoveAction(1)]

    
    actions_by_type = dict()
    actions_by_type["move"] = [
        (u1, MoveAction(1)), (u2, MoveAction(1))
    ]
    env._handle_movement_actions(actions_by_type)
    assert u2.unit_id not in env.state.units["player_0"]
    assert u1.unit_id in env.state.units["player_0"]
    assert u1.power == 10 - math.ceil(5 / 2)

def test_heavy_enter_with_light_entering():
    env: LuxAI_S2 = LuxAI_S2()
    obs = env.reset(seed=0)
    np.random.seed(0)
    while env.state.real_env_steps < 0:
        if env.env_steps == 0:
            actions = dict()
            for p in env.agents:
                actions[p] = dict(bid=0, faction="AlphaStrike")
            obs, _, _, _ = env.step(actions)
        else:
            actions = dict()
            for p in env.agents:
                if my_turn_to_place_factory(env.state.teams[p].place_first, env.state.env_steps):
                    action = place_near_random_ice(p, obs[p])
                else:
                    action = dict()
                actions[p] = action
            obs, _, _, _ = env.step(actions)
        # env.render()
        # time.sleep(0.5)
    
    # add heavy
    u1 = env.add_unit(env.state.teams["player_0"], UnitType.HEAVY, np.array([20, 10]))
    u1.power = 10
    u1.action_queue = [MoveAction(1)]
    u2 = env.add_unit(env.state.teams["player_0"], UnitType.HEAVY, np.array([20, 10]))
    u2.power = 5
    u2.action_queue = [MoveAction(1)]

    light1 = env.add_unit(env.state.teams["player_0"], UnitType.LIGHT, np.array([20, 10]))
    light1.power = 30
    light1.action_queue = [MoveAction(1)]

    
    actions_by_type = dict()
    actions_by_type["move"] = [
        (u1, MoveAction(1)), (u2, MoveAction(1)), (light1, MoveAction(1))
    ]
    env._handle_movement_actions(actions_by_type)
    assert u2.unit_id not in env.state.units["player_0"]
    assert light1.unit_id not in env.state.units["player_0"]
    assert u1.unit_id in env.state.units["player_0"]
    assert u1.power == 10 - math.ceil(5 / 2)

if __name__ == "__main__":
    test_heavy_entering_only()
    test_heavy_enter_with_light_entering()