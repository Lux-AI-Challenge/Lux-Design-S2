from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

import gym
import jax
import numpy as np
import numpy.typing as npt
from gym import spaces
from jux.env import JuxEnv, JuxEnvBatch
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from stable_baselines3.common.vec_env.util import (
    copy_obs_dict,
    dict_to_obs,
    obs_space_info,
)

import luxai_s2.env
from luxai_s2.env import LuxAI_S2
from luxai_s2.state import ObservationStateDict
from luxai_s2.unit import ActionType, BidActionType, FactoryPlacementActionType
from luxai_s2.utils import my_turn_to_place_factory
from luxai_s2.wrappers.controllers import Controller


class SB3JaxVecEnv(gym.Wrapper, VecEnv):
    """
    Creates a simple vectorized wrapper for multiple environments, calling each environment in sequence on the current
    Python process. This is useful for computationally simple environment such as ``cartpole-v1``,
    as the overhead of multiprocess or multithread outweighs the environment computation time.
    This can also be used for RL methods that
    require a vectorized environment, but that you want a single environments to train with.

    :param env_fns: a list of functions
        that return environments to vectorize
    :raises ValueError: If the same environment instance is passed as the output of two or more different env_fn.
    """

    def __init__(
        self,
        env: JuxEnvBatch,
        num_envs: int,
        bid_policy: Callable[
            [str, ObservationStateDict], Dict[str, BidActionType]
        ] = None,
        factory_placement_policy: Callable[
            [str, ObservationStateDict], Dict[str, FactoryPlacementActionType]
        ] = None,
        controller: Controller = None,
    ):
        gym.Wrapper.__init__(self, env)
        self.env = env

        assert controller is not None

        # set our controller and replace the action space
        self.controller = controller
        self.action_space = controller.action_space
        dummy_obs_space = spaces.Box(-1, 1, (1,))

        self.observation_space = dummy_obs_space

        VecEnv.__init__(self, num_envs, dummy_obs_space, action_space=self.action_space)

        # self.keys, shapes, dtypes = obs_space_info(obs_space)

        # self.buf_obs = OrderedDict([(k, np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k])) for k in self.keys])
        # self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        # self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        # self.buf_infos = [{} for _ in range(self.num_envs)]
        # self.actions = None
        self.metadata = {}

    def step_async(self, actions: np.ndarray) -> None:
        self._async_actions = actions

    def step_wait(self):  # noqa: D102
        return self.step(self._async_actions)

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        seeds = []
        for idx, env in enumerate(self.envs):
            seeds.append(env.seed(seed + idx))
        return seeds

    def reset(self, **kwargs) -> VecEnvObs:
        # we upgrade the reset function here

        if "seed" in kwargs:
            seed = kwargs["seed"]
            jax

        # we call the original reset function first
        obs = self.env.reset(**kwargs)

        # then use the bid policy to go through the bidding phase
        action = dict()
        for agent in self.env.agents:
            action[agent] = self.bid_policy(agent, obs[agent])
        obs, _, _, _ = self.env.step(action)

        # while real_env_steps < 0, we are in the factory placement phase
        # so we use the factory placement policy to step through this
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
            obs, _, _, _ = self.env.step(action)
        self.prev_obs = obs

        return obs

    def close(self) -> None:
        return

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        return

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        raise NotImplementedError

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:
        raise NotImplementedError

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs
    ) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [
            getattr(env_i, method_name)(*method_args, **method_kwargs)
            for env_i in target_envs
        ]

    def env_is_wrapped(
        self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs]

    def _get_target_envs(self, indices: VecEnvIndices) -> List[gym.Env]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]
