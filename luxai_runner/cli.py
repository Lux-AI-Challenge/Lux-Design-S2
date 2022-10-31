import asyncio
from typing import Dict, List
from luxai_runner.bot import Bot
from luxai2022 import LuxAI2022
import numpy as np
import json
from luxai_runner.episode import Episode, EpisodeConfig, ReplayConfig

from luxai_runner.logger import Logger
from omegaconf import OmegaConf

def main():
    np.random.seed(0)
    import argparse

    parser = argparse.ArgumentParser(description="Run the LuxAI 2022 game.")
    parser.add_argument("players", nargs="+", help="Paths to player modules.")
    parser.add_argument("-l", "--len", help="Max episode length", type=int, default=1000)

    # replay configs
    parser.add_argument("-o", "--output", help="Output file")
    parser.add_argument("--replay.save_format", help="Save format \"json\" works with the visualizer while pickle is a compact, python usable version", default="json")
    parser.add_argument("--replay.compressed_obs", help="Whether to save compressed observations or not. Compressed observations do not contain the full observation at each step. In particular, the map information is stored as the first observation, subsequent observations only store the changes that happened.", default=True)

    # episode configs
    parser.add_argument(
        "-v", "--verbose", help="Verbose Level (0 = silent, 1 = errors, 2 = warnings, 3 = info)", type=int, default=1
    )
    parser.add_argument("-s", "--seed", help="Random seed for episode(s)", type=int)

    # env configs

    parser.add_argument("--render", help="Render...", action="store_true", default=False)

    # parser.add_argument("--tournament", type=bool)

    args = parser.parse_args()

    # TODO make a tournament runner ranked by ELO, Wins/Losses, Trueskill, Bradley-Terry system
    cfg = EpisodeConfig(
            players=args.players,
            env_cls=LuxAI2022,
            seed=args.seed,
            env_cfg=dict(
                verbose=args.verbose,
                validate_action_space=True,
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
    print(cfg)
    eps = Episode(
        cfg=cfg
    )
    asyncio.run(eps.run())

if __name__ == "__main__":
    main()