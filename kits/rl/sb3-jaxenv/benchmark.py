"""
Implementation of RL agent. Note that luxai_s2 and stable_baselines3 are packages not available during the competition running (ATM)
"""


import copy
import os.path as osp

import gym
import numpy as np
from jux.config import EnvConfig, JuxBufferConfig
from jux.env import JuxEnv
from heuristics.factory import place_factory_near_random_ice
from wrappers import SimpleUnitDiscreteController, SimpleUnitObserver
from wrappers.sb3jax import SB3JaxVecEnv

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Simple benchmarking script"
    )
    parser.add_argument("-s", "--seed", type=int, default=12, help="seed for training")
    parser.add_argument(
        "-n",
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel envs to run. Note that the rollout size is configured separately and invariant to this value",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=200,
        help="Max steps per episode before truncating them",
    )
    args = parser.parse_args()
    return args


def make_env(seed: int = 0, max_episode_steps=100, num_envs=4):
    MAX_N_UNITS = 20
    env_cfg = EnvConfig(MAX_FACTORIES=2)
    jux_env = JuxEnv(
        env_cfg=env_cfg,
        buf_cfg=JuxBufferConfig(MAX_N_UNITS=MAX_N_UNITS),
    )
    env = SB3JaxVecEnv(
        jux_env,
        num_envs=num_envs,
        factory_placement_policy=place_factory_near_random_ice,
        controller=SimpleUnitDiscreteController(jux_env),
        observer=SimpleUnitObserver(),
        max_episode_steps=max_episode_steps,
    )
    env.reset(seed=seed)
    return env

import time


def main(args):
    print(f"=== Compiling + Warmup ===")
    stime = time.time()
    env = make_env(
        args.seed,
        max_episode_steps=args.max_episode_steps,
        num_envs=args.n_envs,
    )
    env.reset()
    env.step(
        dict(player_0=np.zeros(args.n_envs), player_1=np.zeros(args.n_envs))
    )
    print(f"Compile + Warmup Time: {time.time() - stime}")
    env.reset()
    stime = time.time()
    
    
    rounds = 4
    N = args.max_episode_steps * rounds
    for _ in range(N):
        env.step(
            dict(player_0=np.zeros(args.n_envs) + 1, player_1=np.zeros(args.n_envs))
        )
    etime = time.time()
    total_frames = N * args.n_envs
    print(f"FPS {(total_frames / (etime - stime)):.4f}. Frames {total_frames}. One Episode Time: {(etime - stime) / rounds:.4f}s")
    exit()

if __name__ == "__main__":
    main(parse_args())
