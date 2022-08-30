import asyncio
from typing import Dict, List
from luxai_runner.bot import Bot
from luxai2022 import LuxAI2022
import numpy as np
import json
from luxai_runner.episode import Episode, EpisodeConfig

from luxai_runner.logger import Logger


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

    # parser.add_argument("--tournament", type=bool)

    args = parser.parse_args()
    
    # TODO make a tournament runner ranked by ELO, Wins/Losses, Trueskill, Bradley-Terry system
    eps = Episode(cfg=EpisodeConfig(players=args.players, env_cls=LuxAI2022, seed=0, env_cfg=dict(
        verbose=args.verbose,
        validate_action_space=True,
    ),verbosity=args.verbose))
    asyncio.run(eps.run())