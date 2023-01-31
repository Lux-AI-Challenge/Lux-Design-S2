import copy
import os.path as osp

import gym
import numpy as np
import torch as th
import torch.nn as nn
from gym import spaces
from gym.wrappers import TimeLimit
from luxai_s2.state import (ObservationStateDict, StatsStateDict,
                            create_empty_stats)
from luxai_s2.utils.heuristics.factory import build_single_heavy
from luxai_s2.utils.heuristics.factory_placement import place_near_random_ice
from luxai_s2.wrappers import (SB3Wrapper, SimpleUnitDiscreteController,
                               SimpleUnitObservationWrapper)
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecCheckNan, VecVideoRecorder)
from stable_baselines3.ppo import PPO


class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        """
        Adds a custom reward and turns the LuxAI_S2 environment into a single-agent environment for easy training
        """
        super().__init__(env)

    def step(self, action):
        agent = "player_0"
        opp_agent = "player_1"

        opp_factories = self.env.state.factories[opp_agent]
        for k in opp_factories:
            factory = opp_factories[k]
            factory.cargo.water = 1000  # set enemy factories to have 1000 water to keep them alive the whole around and treat the game as single-agent

        action = {agent: action}
        obs, reward, done, info = super().step(action)

        # this is the observation seen by both agents
        shared_obs: ObservationStateDict = self.env.prev_obs[agent]
        done = done[agent]

        # we collect stats on teams here:
        stats: StatsStateDict = self.env.state.stats[agent]

        # compute reward
        # we simply want to encourage the heavy units to move to ice tiles
        # and mine them and then bring them back to the factory and dump it
        # as well as survive as long as possible

        factories = shared_obs["factories"][agent]
        factory_pos = None
        for unit_id in factories:
            factory = factories[unit_id]
            # note that ice converts to water at a 4:1 ratio
            factory_pos = np.array(factory["pos"])
            break
        units = shared_obs["units"][agent]
        unit_deliver_ice_reward = 0
        unit_move_to_ice_reward = 0
        unit_overmining_penalty = 0
        penalize_power_waste = 0

        ice_map = shared_obs["board"]["ice"]
        ice_tile_locations = np.argwhere(ice_map == 1)

        def manhattan_dist(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        unit_power = 0
        for unit_id in units.keys():
            unit = units[unit_id]
            if unit["unit_type"] == "HEAVY":
                pos = np.array(unit["pos"])
                ice_tile_distances = np.mean((ice_tile_locations - pos) ** 2, 1)
                closest_ice_tile = ice_tile_locations[np.argmin(ice_tile_distances)]
                dist_to_ice = manhattan_dist(closest_ice_tile, pos)
                unit_power = unit["power"]
                if unit["cargo"]["ice"] < 20:

                    dist_penalty = min(
                        1.0, dist_to_ice / (10)
                    )  # go beyond 12 squares manhattan dist and no reward
                    unit_move_to_ice_reward += (
                        1 - dist_penalty
                    ) * 0.1  # encourage unit to move to ice
                else:
                    if factory_pos is not None:
                        dist_to_factory = manhattan_dist(pos, factory_pos)
                        dist_penalty = min(1.0, dist_to_factory / 10)
                        unit_deliver_ice_reward = (
                            0.2 + (1 - dist_penalty) * 0.1
                        )  # encourage unit to move back to factory
                # if action[agent]
                # if action[agent] == 15 and unit["power"] < 70:
                #     # penalize the agent for trying to dig with insufficient power, which wastes 10 power for trying to update the action queue
                #     penalize_power_waste -= 0.005

        # save some stats to the info object so we can record it with our SB3 logger
        info = dict()
        metrics = dict()
        metrics["ice_dug"] = (
            stats["generation"]["ice"]["HEAVY"] + stats["generation"]["ice"]["LIGHT"]
        )
        metrics["water_produced"] = stats["generation"]["water"]
        metrics["action_queue_updates_success"] = stats["action_queue_updates_success"]
        metrics["action_queue_updates_total"] = stats["action_queue_updates_total"]

        metrics["unit_deliver_ice_reward"] = unit_deliver_ice_reward
        metrics["unit_move_to_ice_reward"] = unit_move_to_ice_reward

        info["metrics"] = metrics

        reward = (
            0
            + unit_move_to_ice_reward
            + unit_deliver_ice_reward
            + unit_overmining_penalty
            + metrics["water_produced"] / 10
            + penalize_power_waste
        )
        reward = reward

        return obs["player_0"], reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)["player_0"]
        self.prev_step_metrics = None
        return obs


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Simple script that simplifies Lux AI Season 2 as a single-agent environment with a reduced observation and action space. It trains a policy that can succesfully control a heavy unit to dig ice and transfer it back to a factory to keep it alive"
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
        default=100,
        help="Max steps per episode before truncating them",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=3_000_000,
        help="Total timesteps for training",
    )

    parser.add_argument(
        "--eval",
        action="store_true",
        help="If set, will only evaluate a given policy. Otherwise enters training mode",
    )
    parser.add_argument(
        "--model-path", type=str, help="Path to SB3 model weights to use for evaluation"
    )
    parser.add_argument(
        "-l",
        "--log-path",
        type=str,
        default="logs",
        help="Logging path",
    )
    args = parser.parse_args()
    return args


def make_env(env_id: str, rank: int, seed: int = 0, max_episode_steps=100):
    def _init() -> gym.Env:
        # verbose = 0
        # collect stats so we can create reward functions
        # max factories set to 2 for simplification and keeping returns consistent as we survive longer if there are more initial resources
        env = gym.make(env_id, verbose=0, collect_stats=True, MAX_FACTORIES=2)

        # Add a SB3 wrapper to make it work with SB3 and simplify the action space with the controller
        # this will remove the bidding phase and factory placement phase. For factory placement we use
        # the provided place_near_random_ice function which will randomly select an ice tile and place a factory near it.
        env = SB3Wrapper(
            env,
            factory_placement_policy=place_near_random_ice,
            controller=SimpleUnitDiscreteController(env.env_cfg, max_robots=1),
        )
        env = SimpleUnitObservationWrapper(
            env, max_robots=1
        )  # changes observation to include a few simple features
        env = CustomEnvWrapper(env)  # convert to single agent and add our reward
        env = TimeLimit(
            env, max_episode_steps=max_episode_steps
        )  # set horizon to 100 to make training faster. Default is 1000
        env = Monitor(env)  # for SB3 to allow it to record metrics
        env.reset(seed=seed + rank)
        set_random_seed(seed)
        return env

    return _init


class TensorboardCallback(BaseCallback):
    def __init__(self, tag: str, verbose=0):
        super().__init__(verbose)
        self.tag = tag

    def _on_step(self) -> bool:
        c = 0

        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]
                c += 1
                for k in info["metrics"]:
                    stat = info["metrics"][k]
                    self.logger.record_mean(f"{self.tag}/{k}", stat)
        return True


def evaluate(args, env_id, model):
    model = model.load(args.model_path)
    video_length = 1000  # default horizon
    eval_env = SubprocVecEnv(
        [make_env(env_id, i, max_episode_steps=1000) for i in range(args.n_envs)]
    )
    eval_env = VecVideoRecorder(
        eval_env,
        osp.join(args.log_path, "eval_videos"),
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix=f"evaluation_video",
    )
    eval_env.reset()
    out = evaluate_policy(model, eval_env, render=False, deterministic=False)
    print(out)


def train(args, env_id, model: PPO):
    eval_env = SubprocVecEnv(
        [make_env(env_id, i, max_episode_steps=1000) for i in range(4)]
    )
    video_length = 1000
    eval_env = VecVideoRecorder(
        eval_env,
        osp.join(args.log_path, "eval_videos"),
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix=f"evaluation-{env_id}",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=osp.join(args.log_path, "models"),
        log_path=osp.join(args.log_path, "eval_logs"),
        eval_freq=24_000,
        deterministic=False,
        render=False,
    )
    model.learn(
        args.total_timesteps,
        callback=[TensorboardCallback(tag="train_metrics"), eval_callback],
    )
    model.save(args.log_path, "latest_model")


def main(args):
    print("Training with args", args)
    if args.seed is not None:
        set_random_seed(args.seed)
    env_id = "LuxAI_S2-v0"
    env = SubprocVecEnv(
        [
            make_env(env_id, i, max_episode_steps=args.max_episode_steps)
            for i in range(args.n_envs)
        ]
    )
    env.reset()
    rollout_steps = 4_000
    policy_kwargs = dict(net_arch=(128, 128))
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=rollout_steps // args.n_envs,
        batch_size=1000,
        learning_rate=3e-4,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_epochs=3,
        target_kl=0.07,
        gamma=0.97,
        tensorboard_log=osp.join(args.log_path),
    )
    if args.eval:
        evaluate(args, env_id, model)
    else:
        train(args, env_id, model)


if __name__ == "__main__":
    # python ../examples/sb3.py -l logs/exp_1 -s 42 -n 1
    main(parse_args())
