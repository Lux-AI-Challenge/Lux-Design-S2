import time

import numpy as np
from luxai2022.env import LuxAI2022
if __name__ == "__main__":
    env: LuxAI2022 = LuxAI2022()
    o = env.reset()
    env.render()
    time.sleep(0.5)
    # u = Unit(team=Team(1, FactionTypes.MotherMars), unit_type=UnitType.HEAVY, unit_id='1s')
    # env.state.units[1].append(u)
    # observation, reward, done, info = env.last()
    o, r, d, _ = env.step(
        {
            "player_0": dict(faction="MotherMars", spawns=np.array([[4, 4], [15, 5]])),
            "player_1": dict(faction="AlphaStrike", spawns=np.array([[56, 55], [40, 42]])),
        }
    )
    env.render()
    # print(o, r, d)
    s_time = time.time_ns()
    N = 10000
    import ipdb
    for i in range(N):
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
            if i == 0:
                for unit_id, factory in factories.items():
                    actions[unit_id] = 0
            for unit_id, unit in obs["units"][agent].items():
                # actions[unit_id] = np.array([0, np.random.randint(5), 0, 0, 0])
                # make units go to 0, 0
                pos = unit['pos']
                target_pos = np.array([32, 32])
                diff = target_pos - pos
                # print(pos, diff)
                direc = 0
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
                actions[unit_id] = np.array([0, direc, 0, 0, 0])
            all_actions[agent] = actions
        # ipdb.set_trace()
        # env.action_space("player_0").sample()
        # print(all_actions)
        # all_actions["player_0"] = all_actions["player_1"]
        o, r, d, _ = env.step(all_actions)
        for agent in env.agents:
            for unit in env.state.units[agent].values():
                unit.power = 100
        if np.all([d[k] for k in d]):
            o = env.reset()
            env.render()
            print(f"=== {i} ===")
        env.render()
        time.sleep(0.1)
    e_time = time.time_ns()
    print(f"FPS={N / ((e_time - s_time) * 1e-9)}")