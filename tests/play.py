import numpy as np
from luxai2022.env import LuxAI2022
from luxai2022.replay.replay import generate_replay
import copy
if __name__ == "__main__":
    import time
    np.random.seed(0)

    env: LuxAI2022 = LuxAI2022(verbose=0, validate_action_space=False)
    o = env.reset()
    render = True
    if render: 
        env.render()
        time.sleep(0.1)
    states = [env.state.get_compressed_obs()]
    
    # u = Unit(team=Team(1, FactionTypes.MotherMars), unit_type=UnitType.HEAVY, unit_id='1s')
    # env.state.units[1].append(u)
    # observation, reward, done, info = env.last()
    
    # env.render()
    # print(o, r, d)
    s_time = time.time_ns()
    foward_pass_time = 0
    N = 2000
    step = 0
    import ipdb
    for i in range(N):
        if step == 0:
            o, r, d, _ = env.step(
                {
                    "player_0": dict(faction="MotherMars", spawns=np.array([[4, 4], [15, 5]])),
                    "player_1": dict(faction="AlphaStrike", spawns=np.array([[56, 55], [40, 42]])),
                }
            )
        all_actions = dict()
        for team_id, agent in enumerate(env.possible_agents):
            obs = o[agent]
            all_actions[agent] = dict()
            # units = o[agent]["units"]
            # actions = []
            # for unit_id, unit in units.items():
            #     actions.append(dict(unit_id=unit_id))
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
            all_actions[agent] = actions
        
        o, r, d, _ = env.step(all_actions)
        step += 1
        
        states += [env.state.get_compressed_obs()]
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
    replay=to_json(dict(states=states))
    with open("replay.json", "w") as f:
        json.dump(replay, f)
