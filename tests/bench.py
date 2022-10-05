### pyinstrument tests/play.py
import numpy as np
from luxai2022.env import LuxAI2022
from luxai2022.replay.replay import generate_replay
import copy
spawns = None
def policy(agent, step, obs):
    global spawns
    factory_placement_period = False
    obs = obs[agent]
    if step > 0 and step <= obs["board"]["factories_per_team"] + 1:
        factory_placement_period = True
    if step == 0:
        spawns = obs["board"]["spawns"]
        if agent == "player_0":
            return dict(faction="MotherMars", bid=10)
        else:
            return dict(faction="AlphaStrike", bid=9)
    elif factory_placement_period:
        water_left = obs["teams"][agent]["water"]
        metal_left = obs["teams"][agent]["metal"]
        factories_to_place = obs["teams"][agent]["factories_to_place"]
        if agent == "player_0":
            spawn_loc = spawns[agent][np.random.randint(0, len(spawns[agent]))]
            return dict(spawn=spawn_loc, metal=62, water=62)
        else:
            spawn_loc = spawns[agent][np.random.randint(0, len(spawns[agent]))]
            return dict(spawn=spawn_loc, metal=100, water=100)
    else:
        factories = obs["factories"][agent]
        actions = dict()
        if step % 4 == 0 and step > 1:
            for unit_id, factory in factories.items():
                actions[unit_id] = np.random.randint(0,2)
        else:
            for unit_id, factory in factories.items():
                actions[unit_id] = 2
        for unit_id, unit in obs["units"][agent].items():
            # actions[unit_id] = np.array([0, np.random.randint(5), 0, 0, 0])
            # make units go to 0, 0
            pos = unit['pos']
            target_pos = np.array([32 + np.random.randint(-10, 10), 32 + np.random.randint(-10, 10)])
            diff = target_pos - pos
            # print(pos, diff)
            direc = 0
            if np.random.randint(0, 2) == 0:
                if diff[0] != 0:
                    if diff[0] > 0:
                        direc = 2
                    else:
                        direc = 4
                elif diff[1] != 0:
                    if diff[1] > 0:
                        direc = 3
                    else:
                        direc = 1
            else:
                direc = np.random.randint(0,5)
            actions[unit_id] = []
            for i in range(10):
                actions[unit_id] += [np.array([0, direc, 0, 0, 0])]
        return actions



if __name__ == "__main__":
    import time
    np.random.seed(0)

    env: LuxAI2022 = LuxAI2022(verbose=0, validate_action_space=False)
    o = env.reset(seed=0)
    render = False
    if render: 
        env.render()
        time.sleep(0.1)
    states = [env.state.get_compressed_obs()]
    
    s_time = time.time_ns()
    foward_pass_time = 0
    N = 10000
    step = 0
    for i in range(N):
        all_actions = dict()
        # import ipdb;ipdb.set_trace()
        for team_id, agent in enumerate(env.possible_agents):
            all_actions[agent] = dict()
            all_actions[agent] = policy(agent, step, o)
        
        o, r, d, _ = env.step(all_actions)
        step += 1
        
        # states += [env.state.get_compressed_obs()]
        for agent in env.agents:
            for unit in env.state.units[agent].values():
                unit.power = 100
            for factory in env.state.factories[agent].values():
                factory.power = 1000
                factory.cargo.water = 1000
        if np.all([d[k] for k in d]):
            o = env.reset()
            
            if render: env.render()
            print(f"=== {i} ===")
            e_time = time.time_ns()
            print(f"FPS={step / ((e_time - s_time) * 1e-9)}")
            s_time = time.time_ns()
            step = 0
            break
        if render:
            env.render()
            # time.sleep(0.1)
    # e_time = time.time_ns()
    # print(f"FPS={N / ((e_time - s_time) * 1e-9)}")
    import json
    def to_json(state):
        if isinstance(state, np.ndarray):
            return state.tolist()
        elif isinstance(state, np.int64):
            return state.tolist()
        elif isinstance(state, list):
            return [to_json(s) for s in state]
        elif isinstance(state, dict):
            out = {}
            for k in state:
                out[k] = to_json(state[k])
            return out
        else:
            return state
    # import pickle
    # with open("replay.pkl", "wb") as f:
    #     pickle.dump(dict(states=states), f)

    # states = generate_replay(states)
    # replay = to_json(dict(states=states))
    # import ipdb;ipdb.set_trace()
    # with open("replay.json", "w") as f:
    #     json.dump(to_json(dict(states=states)), f)
