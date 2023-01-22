from typing import Dict

import gym
import numpy as np
from gym import spaces
from gym import RewardWrapper
from stable_baselines3.common.vec_env.base_vec_env import (VecEnv,
                                                           VecEnvStepReturn,
                                                           VecEnvWrapper)
import numpy.typing as npt
import luxai_s2.env
from luxai_s2.env import LuxAI_S2
from luxai_s2.state import ObservationStateDict
from luxai_s2.utils import my_turn_to_place_factory
from luxai_s2.wrappers.controllers import Controller, SimpleDiscreteController


class SB3Wrapper(gym.Wrapper):
    def __init__(
        self,
        env: LuxAI_S2,
        bid_policy=None,
        factory_placement_policy=None,
        controller: Controller=None,
    ) -> None:
        """
        A simple environment wrappr that simplifies the observation and action space

        Action Space Parameterization:
            As on each board tile only one robot can be alive at the start of a turn, a simple way to control all robots
            in a centralized fashion is to predict an action for each board tile.

            The original action parameterization is designed to encapsulate all possible actions. This wrapper will use a
            simpler one. For RL users, we highly recommend reading through this and understanding the code, as wll as making your
            own modifications to help improve RL training.

            In this wrapper the SimpleDiscreteController is used
        """
        gym.Wrapper.__init__(self, env)
        self.env = env
        if controller is None:
            controller = SimpleDiscreteController(self.env.state.env_cfg)
        self.controller = controller

        self.action_space = controller.action_space

        obs_dims = 23 # see _convert_obs function for how this is computed
        self.map_size = self.env.env_cfg.map_size
        self.observation_space = spaces.Box(-999, 999, shape=(self.map_size, self.map_size, obs_dims))

        # The simplified wrapper removes the first two phases of the game by using predefined policies (trained or heuristic)
        # to handle those two phases during each reset
        if factory_placement_policy is None:

            def factory_placement_policy(player, obs):
                potential_spawns = np.array(
                    list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1)))
                )
                spawn_loc = potential_spawns[
                    np.random.randint(0, len(potential_spawns))
                ]
                return dict(spawn=spawn_loc, metal=150, water=150)

        self.factory_placement_policy = factory_placement_policy
        if bid_policy is None:

            def bid_policy(player, obs):
                faction = "AlphaStrike"
                if player == "player_1":
                    faction = "MotherMars"
                return dict(bid=0, faction=faction)

        self.bid_policy = bid_policy

        self._prev_obs = None
        # list of all agents regardless of status
        self.all_agents = []

    def step(self, action: Dict[str, npt.NDArray]):
        lux_action = dict()
        for agent in self.all_agents:
            if agent in action:
                lux_action[agent] = self.controller.action_to_lux_action(agent=agent, obs=self._prev_obs, action=action[agent])
            else:
                lux_action[agent] = dict()
        obs, reward, done, info = self.env.step(lux_action)
        self._prev_obs = obs
        return self._convert_obs(obs), reward, done, info
    def _convert_obs(self, obs: Dict[str, ObservationStateDict]) -> Dict[str, npt.NDArray]:

        shared_obs = obs["player_0"]
        unit_mask = np.zeros((self.map_size, self.map_size, 1))
        unit_data = np.zeros((self.map_size, self.map_size, 9)) # power(1) + cargo(4) + unit_type(1) + unit_pos(2) + team(1)
        factory_mask = np.zeros_like(unit_mask)
        factory_data = np.zeros((self.map_size, self.map_size, 8)) # power(1) + cargo(4) + factory_pos(2) + team(1)
        for agent in self.all_agents:
            factories = shared_obs["factories"][agent]
            units = shared_obs["units"][agent]
            
            for unit_id in units.keys():
                unit = units[unit_id]
                # we encode everything but unit_id or action queue
                cargo_vec = np.array([unit["power"], unit["cargo"]["ice"], unit["cargo"]["ore"], unit["cargo"]["water"], unit["cargo"]["metal"]])
                unit_type = 0 if unit["unit_type"] == "LIGHT" else 1 # note that build actions use 0 to encode Light
                unit_vec = np.concatenate([unit["pos"], [unit_type], cargo_vec, [unit["team_id"]]], axis=-1)

                # note that all data is stored as map[x, y] format
                unit_data[unit["pos"][0], unit["pos"][1]] = unit_vec
                unit_mask[unit["pos"][0], unit["pos"][1]] = 1

            for unit_id in factories.keys():
                factory = factories[unit_id]
                # we encode everything but strain_id or unit_id
                cargo_vec = np.array([factory["power"], factory["cargo"]["ice"], factory["cargo"]["ore"], factory["cargo"]["water"], factory["cargo"]["metal"]])
                factory_vec = np.concatenate([factory["pos"], cargo_vec, [factory["team_id"]]], axis=-1)
                factory_data[factory["pos"][0], factory["pos"][1]] = factory_vec
                factory_mask[factory["pos"][0], factory["pos"][1]] = 1
            
            image_features = np.concatenate(
                [
                    np.expand_dims(shared_obs["board"]["lichen"], -1) / self.env.state.env_cfg.MAX_LICHEN_PER_TILE,
                    np.expand_dims(shared_obs["board"]["rubble"], -1) / self.env.state.env_cfg.MAX_RUBBLE,
                    np.expand_dims(shared_obs["board"]["ice"], -1),
                    np.expand_dims(shared_obs["board"]["ore"], -1),
                    unit_mask,
                    unit_data,
                    factory_mask,
                    factory_data
                
                ],
                axis=-1
            )

        new_obs = dict()
        for agent in self.all_agents:
            new_obs[agent] = image_features
        return new_obs

    def reset(self):
        obs = self.env.reset()
        self.all_agents = self.env.agents
        action = dict()
        for agent in self.all_agents:
            action[agent] = self.bid_policy(agent, obs[agent])
        obs, _, _, _ = self.env.step(action)
        while self.env.state.real_env_steps < 0:
            action = dict()
            for agent in self.all_agents:
                if my_turn_to_place_factory(
                    obs["player_0"]["teams"][agent]["place_first"],
                    self.env.state.env_steps,
                ):
                    action[agent] = self.factory_placement_policy(agent, obs[agent])
                else:
                    action[agent] = dict()
            obs, _, _, _ = self.env.step(action)
        self._prev_obs = obs
        return self._convert_obs(obs)

class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        """
        Add a custom reward and turn the LuxAI_S2 environment into a single-agent form
        by disabling opposition from being destroyed via infinite water and giving them empty actions
        """
        super().__init__(env)
    
    def step(self, action):
        action = dict(player_0=action)
        obs, reward, done, info = super().step(action)
        # define our own reward?
        reward = reward['player_0'] / 100
        done = done['player_0']
        info = info['player_0']
        return obs['player_0'], reward, done, info
    def reset(self):
        return super().reset()['player_0']

if __name__ == "__main__":
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.ppo import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    import torch as th
    import torch.nn as nn
    class CustomCNN(BaseFeaturesExtractor):
        """
        :param observation_space: (gym.Space)
        :param features_dim: (int) Number of features extracted.
            This corresponds to the number of unit for the last layer.
        """

        def __init__(self, observation_space: spaces.Box, features_dim: int = 1024):
            super().__init__(observation_space, features_dim)
            # We assume CxHxW images (channels first)
            # Re-ordering will be done by pre-preprocessing or wrapper
            n_input_channels = observation_space.shape[0]
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 32, kernel_size=5, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )

            # Compute shape by doing one forward pass
            with th.no_grad():
                n_flatten = self.cnn(
                    th.as_tensor(observation_space.sample()[None]).float()
                ).shape[1]
            print("CNN output Shape", n_flatten)
            self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

        def forward(self, observations: th.Tensor) -> th.Tensor:
            return self.linear(self.cnn(observations))
    def make_env(env_id: str, rank: int, seed: int = 0):
        def _init() -> gym.Env:
            env = gym.make(env_id, verbose=0)
            env = SB3Wrapper(env)
            env = CustomEnvWrapper(env)
            env = Monitor(env)
            env.unwrapped.reset(seed=seed + rank)
            return env

        set_random_seed(seed)
        return _init
    set_random_seed(0)
    env_id = "LuxAI_S2-v0"
    num_envs = 8
    # env = make_env(env_id, 0)()
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_envs)])
    obs = env.reset()
    # import ipdb;ipdb.set_trace()
    
    rollout_steps = 8_000
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=1024),
    )
    model = PPO("CnnPolicy", env, n_steps=rollout_steps // num_envs, batch_size=400,  policy_kwargs=policy_kwargs, verbose=1)
    import ipdb;ipdb.set_trace()
    model.learn(10_000_000)
    
    