"""
This file is where your agent's logic is kept. Define a bidding policy, factory placement policy, as well as a policy for playing the normal phase of the game

The tutorial will learn an RL agent to play the normal phase and use heuristics for the other two phases.

Note that like the other kits, you can only debug print to standard error e.g. print("message", file=sys.stderr)
"""

import os.path as osp
import sys

import jax
import jax.numpy as jnp
import numpy as np
import torch as th
from stable_baselines3.ppo import PPO

from heuristics.factory import place_factory_near_random_ice
from lux.config import EnvConfig
from utils import lux_obs_to_lux_state, lux_state_to_jux_state
from wrappers import SimpleUnitDiscreteController, SimpleUnitObservationWrapper

# change this to use weights stored elsewhere
# make sure the model weights are submitted with the other code files
# any files in the logs folder are not necessary
MODEL_WEIGHTS_RELATIVE_PATH = "./best_model"
from jux.env import JuxBufferConfig

buf_cfg = JuxBufferConfig()


class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.player_id = 0  # numerical version is for jax functions
        if self.player == "player_1":
            self.player_id = 1

        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg

        directory = osp.dirname(__file__)

        self.policy = PPO.load(
            osp.join(directory, MODEL_WEIGHTS_RELATIVE_PATH), device="cpu"
        )

        self.controller = SimpleUnitDiscreteController(self.env_cfg)

        self.key = jax.random.PRNGKey(np.random.randint(0, 9999))

    def bid_policy(self, step: int, obs, remainingOverageTime: int = 60):
        return dict(faction="AlphaStrike", bid=0)

    def factory_placement_policy(self, step: int, obs, remainingOverageTime: int = 60):
        can_place = False
        if obs["teams"][self.player]["place_first"]:
            if step % 2 == 1:
                can_place = True
        else:
            if step % 2 == 0:
                can_place = True
        # skip our turn if we can't place now or used up all our metal
        if not can_place:
            return dict()
        if obs["teams"][self.player]["metal"] <= 0:
            return dict()

        key, subkey = jax.random.split(self.key)
        self.key = key
        lux_state = lux_obs_to_lux_state(obs, self.env_cfg)
        state = lux_state_to_jux_state(lux_state, buf_cfg)
        return place_factory_near_random_ice(subkey, self.player_id, state)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        # first convert observations using the same observation wrapper you used for training
        # note that SimpleUnitObservationWrapper takes input as the full observation for both players and returns an obs for players
        raw_obs = dict(player_0=obs, player_1=obs)
        # obs = SimpleUnitObservationWrapper.convert_obs(raw_obs, env_cfg=self.env_cfg)
        # obs = obs[self.player]

        # obs = th.from_numpy(obs).float()
        # with th.no_grad():

        #     # to improve performance, we have a rule based action mask generator for the controller used
        #     # which will force the agent to generate actions that are valid only.
        #     action_mask = (
        #         th.from_numpy(self.controller.action_masks(self.player, raw_obs))
        #         .unsqueeze(0)
        #         .bool()
        #     )

        #     # SB3 doesn't support invalid action masking. So we do it ourselves here
        #     features = self.policy.policy.features_extractor(obs.unsqueeze(0))
        #     x = self.policy.policy.mlp_extractor.shared_net(features)
        #     logits = self.policy.policy.action_net(x) # shape (1, N) where N=12 for the default controller

        #     logits[~action_mask] = -1e8 # mask out invalid actions
        #     dist = th.distributions.Categorical(logits=logits)
        #     actions = dist.sample().cpu().numpy() # shape (1, 1)

        # # use our controller which we trained with in train.py to generate a Lux S2 compatible action
        # lux_action = self.controller.action_to_lux_action(
        #     self.player, raw_obs, actions[0]
        # )

        # # commented code below adds watering lichen which can easily improve your agent
        # # shared_obs = raw_obs[self.player]
        # # factories = shared_obs["factories"][self.player]

        lux_state = lux_obs_to_lux_state(obs, self.env_cfg)
        # print(lux_state.global_id, file=sys.stderr)
        lux_action = dict()
        for unit_id in obs["factories"][self.player].keys():
            factory = obs["factories"][self.player][unit_id]
            # if 1000 - step < 50 and factory["cargo"]["water"] > 100:
            lux_action[unit_id] = 2  # water and grow lichen at the very end of the game
        return lux_action
