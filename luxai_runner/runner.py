import asyncio
from typing import Dict
from luxai_runner.bot import Bot
from luxai2022 import LuxAI2022
import numpy as np
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
async def main(args):
    if len(args.players) != 2:
        raise ValueError("Must provide two paths.")

    players: Dict[str, Bot] = dict()
    for i in range(2):
        player = Bot(args.players[i], f"player_{i}", i, args.verbose)
        players[player.agent] = player
        await asyncio.wait([player.proc.start()], return_when=asyncio.ALL_COMPLETED)
    # exit()

    env: LuxAI2022 = LuxAI2022(verbose=args.verbose, validate_action_space=True)
    seed = args.seed if args.seed is not None else np.random.randint(9999999)
    obs = env.reset(seed=seed)
    game_done = False
    rewards, dones, infos = dict(), dict(), dict()
    for agent in env.agents:
        rewards[agent] = 0 
        dones[agent] = 0
        infos[agent] = dict()

    while not game_done:
        print("===", env.env_steps)
        actions = dict()
        obs = env.state.get_compressed_obs()
        obs = to_json(obs)
        agent_ids = []
        action_coros = []
        for agent in players:
            player = players[agent]
            action = player.step(obs, env.env_steps, rewards[agent], infos[agent])
            action_coros += [action]
            agent_ids += [player.agent]
        resolved_actions = await asyncio.gather(*action_coros)
        for agent_id, action in zip(agent_ids, resolved_actions):
            actions[agent_id] = action
        print(actions)
        obs, rewards, dones, infos = env.step(actions)

        players_left = len(dones)
        for k in dones:
            if dones[k]: players_left -= 1
        if players_left == 0:
            game_done = True
        

    pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the LuxAI 2022 game.")
    parser.add_argument('players', nargs="+", help="Paths to player modules.")
    # parser.add_argument("-r", "--rounds", help="Max rounds in game", type=int, default=2000)
    
    
    parser.add_argument("-o", "--output", help="Output file")
    
    parser.add_argument("-v", "--verbose", help="Verbose Level (0 = silent, 1 = errors, 2 = warnings, 3 = info)", type=int, default=1)
    parser.add_argument("-s", "--seed", help="Random seed for episode(s)", type=int)

    ## Map Arguments
    # TODO None of these are actually being used yet.
    parser.add_argument("-map", "--map_type", help="Map type ('Cave', 'Craters', 'Island', 'Mountain')")
    parser.add_argument("--size", help="Size (32-64)", type=int)
    parser.add_argument("-sym", "--symmetry", help="Symmetry ('horizontal', 'rotational', 'vertical', '/', '\\')")
    args = parser.parse_args()
    asyncio.run(main(args))