# For now, only supporting Python, C++ bots.
# Look into pyv8, pyjnius for js and java.

import numpy as np
from luxai2022.env import LuxAI2022
from luxai2022.replay.replay import generate_replay
import copy

import sys
import importlib.util

def load_bot(path, name, id):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    class Agent(object):
        def __init__(self):
            self.name = name
        def setup(self, observation, configuration):
            return module.spawn(observation, configuration, name)
        def actions(self, observation, configuration):
            return module.agent(observation, configuration, name)

    return Agent()

if __name__ == "__main__":
    import time

    import argparse
    parser = argparse.ArgumentParser(description="Run LuxAI 2022 game.")
    parser.add_argument('players', nargs="+", help="Paths to player modules.")
    parser.add_argument("-r", "--rounds", help="Max rounds in game", type=int, default=2000)
    parser.add_argument("-o", "--output", help="Output file")
    parser.add_argument("-v", "--verbose", help="Verbose Level (0 = silent, 1 = game mechanics, 2 = logs)", type=int, default=1)

    # None of these are actually being used yet.
    parser.add_argument("-t", "--map_type", help="Map type ('Cave', 'Craters', 'Island', 'Mountain')")
    parser.add_argument("-s", "--size", help="Size (32-64)", type=int)
    parser.add_argument("-d", "--seed", help="Seed", type=int)
    parser.add_argument("-m", "--symmetry", help="Symmetry ('horizontal', 'rotational', 'vertical', '/', '\\')")

    args = vars(parser.parse_args())

    if len(args["players"]) != 2:
        raise ValueError("Must provide two paths.")

    # Load players.
    players = dict()
    for i in range(2):
        player = load_bot(args["players"][i], f"player_{i}", i)
        players[player.name] = player

    # Seed game. Right now this seed doesn't actually change the env seed.
    np.random.seed(10)
    if args["seed"]:
        seed = args["seed"]
        np.random.seed(seed)
    else:
        seed = np.random.get_state()[1][0]

    if args["verbose"] > 0:
        print(f"Running game {args['players'][0]} vs {args['players'][1]} on seed {seed}")

    # Initialize environment.
    env: LuxAI2022 = LuxAI2022(verbose=args["verbose"], validate_action_space=True)
    o = env.reset()
    cfg = env.env_cfg

    render = True
    if render:
        env.render()
        time.sleep(0.1)
    states = [env.state.get_compressed_obs()]

    s_time = time.time_ns()
    foward_pass_time = 0

    # Setup factions and spawn locations.
    setup = {}
    for agent in players:
        setup[agent] = players[agent].setup(o[agent], cfg)
    o, r, d, i = env.step(setup)

    for step in range(args["rounds"]):
        if args["verbose"] > 0:
            print(f"=== {step} ===")

        all_actions = dict()
        for team_id, agent in enumerate(env.possible_agents):
            obs = o[agent]
            all_actions[agent] = players[agent].actions(obs, cfg)

        o, r, d, i = env.step(all_actions)

        states += [env.state.get_compressed_obs()]
        if render:
            env.render()

        # Game is over.
        if np.all([d[k] for k in d]):
            o = env.reset()
            e_time = time.time_ns()
            if args["verbose"] > 0:
                print(f"FPS={step / ((e_time - s_time) * 1e-9)}")
            break

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


    replay=to_json(dict(states=states))

    if "output" in args:
        filename = args["output"]
    else:
        from datetime.datetime import now
        t = now()
        filename = "replay/" + "_".join(players) + f"{t.hour}_{t.minute}_{t.second}.json"
    with open(filename, "w") as f:
        json.dump(replay, f)

    if args["verbose"] > 0:
        print(f"Saved replay in {filename}")
