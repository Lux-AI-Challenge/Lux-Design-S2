import gym
import luxai_s2.env
import numpy as np
from gym import spaces
from luxai_s2.env import LuxAI_S2
from luxai_s2.state import State
from luxai_s2.utils import my_turn_to_place_factory
from luxai_s2.wrappers.controllers import (
    Controller,
    SimpleDiscreteController,
)
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvStepReturn,
    VecEnvWrapper,
)


class SB3Wrapper(gym.Wrapper):
    def __init__(
        self,
        env: LuxAI_S2,
        bid_policy=None,
        factory_placement_policy=None,
        controller: Controller = SimpleDiscreteController(48, 48),
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

        self.action_space = gym.spaces.Box(0, 4, shape=(48, 48, 8))
        self.observation_space = spaces.Dict(
            dict(
                image=spaces.Box(0, 4, shape=(48, 48, 8)),
                state=spaces.Box(0, 4, shape=(7,)),
            )
        )
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

    def _convert_obs(self, obs):
        """
        Convert JuxEnv states into a usable observation
        """
        # unit_features = jnp.zeros_like()

        # Factory Features
        # We encode a 4 dim vector [factory_here, ]
        # both teams get the same observation
        obs = obs["player_0"]
        image_features = np.hstack(
            [
                obs["board"]["lichen"][..., None],
                obs["board"]["rubble"][..., None],
                obs["board"]["ice"][..., None],
                obs["board"]["ore"][..., None],
            ]
        )
        obs = dict()
        for agent in self.env.agents:
            obs[agent] = image_features
        return obs

    def reset(self):
        obs = self.env.reset()

        action = dict()
        for agent in self.env.agents:
            action[agent] = self.bid_policy(agent, obs[agent])
        obs, _, _, _ = self.env.step(action)
        while self.env.state.real_env_steps < 0:
            action = dict()
            for agent in self.env.agents:
                if my_turn_to_place_factory(
                    obs["player_0"]["teams"][agent]["place_first"],
                    self.env.state.env_steps,
                ):
                    action[agent] = self.factory_placement_policy(agent, obs[agent])
                else:
                    action[agent] = dict()
            obs, reward, done, info = self.env.step(action)
        return self._convert_obs(obs), reward, done, info


if __name__ == "__main__":
    from stable_baselines3.common.utils import set_random_seed

    def make_env(env_id: str, rank: int, seed: int = 0):
        def _init() -> gym.Env:
            env = gym.make(env_id)
            env.seed(seed + rank)
            return env

        set_random_seed(seed)
        return _init

    env = gym.make("LuxAI_S2-v0")
    env = SB3Wrapper(env)
    obs = env.reset()
    obs = obs["player_0"]
    import ipdb

    ipdb.set_trace()
