import asyncio
from typing import Dict, List
from luxai_runner.bot import Bot
from luxai2022 import LuxAI2022
import numpy as np
import json
from luxai_runner.episode import Episode, EpisodeConfig, ReplayConfig
from luxai_runner.tournament import Tournament, TournamentConfig
from luxai_runner.logger import Logger
from omegaconf import OmegaConf
import sys

def main():
    np.random.seed(0)
    import argparse

    parser = argparse.ArgumentParser(description="Run the LuxAI 2022 game.")
    parser.add_argument("players", nargs="+", help="Paths to player modules. If --tournament is passed as well, you can also pass a folder and we will look through all sub-folders for valid agents with main.py files (only works for python agents at the moment).")
    parser.add_argument("-l", "--len", help="Max episode length", type=int, default=1000)

    # replay configs
    parser.add_argument("-o", "--output", help="Where to output replays. Default is none and no replay is generated")
    parser.add_argument("--replay.save_format", help="Save format \"json\" works with the visualizer while pickle is a compact, python usable version", default="json")
    parser.add_argument("--replay.compressed_obs", help="Whether to save compressed observations or not. Compressed observations do not contain the full observation at each step. In particular, the map information is stored as the first observation, subsequent observations only store the changes that happened.", default=True)

    # episode configs
    parser.add_argument(
        "-v", "--verbose", help="Verbose Level (0 = silent, 1 = errors, 2 = warnings, 3 = info)", type=int, default=1
    )
    parser.add_argument("-s", "--seed", help="Fix a seed for episode(s). All episodes will initialize the same, including tournament ones", type=int)

    # env configs

    parser.add_argument("--render", help="Render with a window", action="store_true", default=False)

    parser.add_argument("--tournament", help="Turn tournament mode on", action="store_true", default=False)
    parser.add_argument("--tournament_cfg.concurrent", help="Max concurrent number of episodes to run. Recommended to set no higher than the number of CPUs / 2", type=int, default=1)
    parser.add_argument("--tournament_cfg.ranking_system", help="The ranking system to use. Default is 'elo'. Can be 'elo', 'wins'.", type=str, default="elo")
    parser.add_argument("--skip_validate_action_space", help="Set this for a small performance increase. Note that turning this on means the engine assumes your submitted actions are valid. If your actions are not well formatted there could be errors", action="store_true", default=False)


    args = parser.parse_args()

    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    cfg = EpisodeConfig(
            players=args.players,
            env_cls=LuxAI2022,
            seed=args.seed,
            env_cfg=dict(
                verbose=args.verbose,
                validate_action_space=not args.skip_validate_action_space,
                max_episode_length=args.len,
            ),
            verbosity=args.verbose,
            save_replay_path=args.output,
            replay_options=ReplayConfig(
                save_format=getattr(args, "replay.save_format"),
                compressed_obs=getattr(args, "replay.compressed_obs")
            ),
            render=args.render
        )

    if args.tournament:
        import os
        if os.path.isdir(args.players[0]):
            assert len(args.players) == 1, "Found more than one positional argument despite being given a directory of players"
            subfolders = [ f.path for f in os.scandir(args.players[0]) if f.is_dir() ]
            agents = []
            for sub_dir in subfolders:
                agent_file = os.path.join(sub_dir, "main.py")
                if os.path.isfile(agent_file):
                    agents.append(agent_file)
            print(f"Found {len(agents)} in {args.players[0]}")
            args.players = agents

        tournament_config = TournamentConfig()
        tournament_config.agents = args.players
        # TODO - in future replace this with OmegaConf or something that can parse these nicely
        tournament_config.max_concurrent_episodes = getattr(args, "tournament_cfg.concurrent")
        tournament_config.ranking_system = getattr(args, "tournament_cfg.ranking_system")
        tourney = Tournament(
            cfg=tournament_config,
            episode_cfg=cfg # the base/default episode config
        )
        # import ipdb;ipdb.set_trace()
        asyncio.run(tourney.run())
        # exit()
    else:
        import time
        stime = time.time()
        eps = Episode(
            cfg=cfg
        )
        asyncio.run(eps.run())
        etime = time.time()
        print(etime - stime)

if __name__ == "__main__":
    main()