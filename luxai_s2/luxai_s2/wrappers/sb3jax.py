from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union, Tuple

import gym
import jax
import numpy as np
import numpy.typing as npt
from gym import spaces
import jax.numpy as jnp
from jux.env import JuxEnv, JuxEnvBatch, JuxAction
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
from chex import Array

import luxai_s2.env
from luxai_s2.env import LuxAI_S2
from luxai_s2.state import ObservationStateDict
from luxai_s2.unit import ActionType, BidActionType, FactoryPlacementActionType
from luxai_s2.utils import my_turn_to_place_factory
from luxai_s2.wrappers.controllers import Controller

from jux.state import State as JuxState


class SB3JaxVecEnv(gym.Wrapper, VecEnv):
    """
    Jax based environment for Stable Baselines 3

    Converts Lux S2 into a single phase game by upgrading the reset function to handle the bidding and factory placement phase in it

    Note that different to the CPU SB3Wrapper bid_policy and factory_placement_policy have different signatures. Namely,
    they accept a jax key as the first input, then the team id (0 or 1), and a Jux State as the 2nd input.

    Finally, the returned actions are pure Arrays. See TODO for how the actions are formatted. TODO alternatively you can submit lux format actions 
    and use TODO utility to convert them
    """

    def __init__(
        self,
        env: JuxEnv,
        num_envs: int,
        bid_policy: Callable[
            [jax.random.KeyArray, int, JuxState], Tuple[Array, Array]
        ] = None,
        factory_placement_policy: Callable[
            [jax.random.KeyArray, int, JuxState], Tuple[Array, Array, Array]
        ] = None,
        controller: Controller = None
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

        self.metadata = {}

        if factory_placement_policy is None:
            def factory_placement_policy(key, player, state):
                spawn_loc = jax.random.randint(key, (2,), 0, self.env.env_cfg.map_size, dtype=jnp.int8)
                return dict(spawn=spawn_loc, metal=150, water=150)

        self.factory_placement_policy = factory_placement_policy
        if bid_policy is None:

            def bid_policy(key, player, state):
                faction = 0
                if player == 0:
                    faction = 1
                return dict(bid=0, faction=faction)

        self.bid_policy = bid_policy
        self.factory_placement_policy = factory_placement_policy


        # create a upgraded reset function that replaces the bid and placement phase
        # it also handles the variable length in episodes by a while loop until real_env_steps is no longer < 0
        def _upgraded_reset(seed: int) -> Tuple[JuxState, Tuple[Dict, int, bool, Dict]]:
            state: JuxState = self.env.reset(seed)
            key, subkey = jax.random.split(state.rng_state)

            bids, factions = jnp.zeros(2, dtype=jnp.int32 ), jnp.zeros(2, dtype=jnp.int32 )
            for i in range(2):
                act = self.bid_policy(subkey, i, state)
                bids = bids.at[i].set(act["bid"])
                factions = factions.at[i].set (act["faction"])
            state, _  = self.env.step_bid(state, bids, factions)

            def body_fun(val):
                state, key = val
                key, subkey = jax.random.split(key)
                
                spawns, waters, metals = jnp.zeros((2, 2), dtype=jnp.int32), jnp.zeros(2, dtype=jnp.int32), jnp.zeros(2, dtype=jnp.int32)
                act = self.factory_placement_policy(subkey, state.next_player, state)
                spawns = spawns.at[state.next_player].set(act["spawn"])
                waters = waters.at[state.next_player].set(act["water"])
                metals = metals.at[state.next_player].set(act["metal"])
                state, (observations, _, _, _) = self.env.step_factory_placement(state, spawns, waters, metals)
                return (state, key)
            def cond_fun(val):
                state, key = val
                return state.real_env_steps < 0
            (state, key) = jax.lax.while_loop(cond_fun, body_fun, (state, key))

            return {'player_0': state, 'player_1': state}


        self._upgraded_reset = jax.vmap(jax.jit(_upgraded_reset))

        self._step_normal_phase = jax.vmap(jax.jit(self.env.step_late_game))

        self.states: JuxState = None
        self.key = jax.random.PRNGKey(np.random.randint(0, 2 ** 16 - 1, dtype=np.int32))

        self.action_to_jux_action = jax.vmap(self.controller.action_to_jux_action)

    def step(self, actions: Dict[str, npt.NDArray]) -> VecEnvStepReturn:
        """
        Steps through the jax env. First using the controller converts actions into lux actions?

        Then converts all to jax actions? Slow???

        actions: Dict[str, (B, N)]
        """


        # create an empty action
        jux_action = JuxAction.empty(
            self.env.env_cfg, 
            self.env.buf_cfg
        )

        # Get the actions of both teams in the JuxAction format
        jux_action_0 = self.action_to_jux_action(jnp.zeros(self.num_envs,dtype=jnp.int8), self.states, actions["player_0"])
        jux_action_1 = self.action_to_jux_action(jnp.ones(self.num_envs, dtype=jnp.int8), self.states, actions["player_1"])
        
        jux_action: JuxAction = jax.tree_map(lambda x : x[None].repeat(self.num_envs, axis=0), jux_action)
        # now every leaf of jux_action is shape (B, 2, ....)
        # jux_action = jax.tree_map(lambda x : x.at[:, 0].set(), jux_action)
        
        def update_team_actions(x,y, team):
            # stack along team dimension
            return jnp.stack([x[:, team], y[:, team]], 1)
        jux_action: JuxAction = jax.tree_map(lambda x, y : update_team_actions(x, y, 0), jux_action, jux_action_0)
        jux_action: JuxAction = jax.tree_map(lambda x, y : update_team_actions(x, y, 1), jux_action, jux_action_1)
        state, (observations, rewards, dones, infos) = self._step_normal_phase(self.states, jux_action)
        self.states = state

        return observations, rewards, dones, infos


    def step_async(self, actions: np.ndarray) -> None:
        self._async_actions = actions

    def step_wait(self):
        return self.step(self._async_actions)

    def seed(self):
        raise NotImplementedError

    def reset(self, **kwargs) -> VecEnvObs:
        if "seed" in kwargs:
            seed = kwargs["seed"]
            self.key = jax.random.PRNGKey(seed=seed)
        
        key, subkey = jax.random.split(self.key)
        self.key = key
        # we call the original reset function first
        seeds = jax.random.randint(subkey, shape=(self.num_envs, ), minval=0, maxval=2**16 - 1, dtype=jnp.int32)
        states_for_players: Dict[str, JuxState] = self._upgraded_reset(seed=seeds)
        self.states = states_for_players["player_0"]
        return {'player_0': self.states, 'player_1': self.states}

    def close(self) -> None:
        return

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        state = jax.tree_map(lambda x: x[0], self.states)
        self.env.render(state)

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        raise NotImplementedError

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:
        raise NotImplementedError

    def env_is_wrapped(
        self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        raise NotImplementedError
    
    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):  # noqa: D102
        raise NotImplementedError