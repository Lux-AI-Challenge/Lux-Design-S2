from typing import Any, Dict, TypeVar

import gym
import numpy as np
import numpy.typing as npt
from gym import spaces
import jax
import jax.numpy as jnp
try:
    # we try/except the jux package as its not installed on Kaggle-Environments
    import jux
    from jux.state import State
except:
    # State = TypeVar("State")
    pass 

class SimpleUnitObservationWrapper(gym.vector.VectorEnvWrapper):
    """
    A simple state based observation to work with in pair with the SimpleUnitDiscreteController

    It contains info only on the first robot, the first factory you own, and some useful features. If there are no owned robots the observation is just zero.
    No information about the opponent is included. This will generate observations for all teams.

    Included features:
    - First robot's stats
    - distance vector to closest ice tile
    - distance vector to first factory

    """

    def __init__(self, env) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(-999, 999, shape=(11,))

    def observation(self, obs):
        return SimpleUnitObservationWrapper.convert_jux_obs(obs)

    @staticmethod
    def convert_jux_obs(state: Dict[str, State]) -> Dict[str, npt.NDArray]:
        # converts states into batch observations
        # we know that the jax state of lux always has units and factories at the front of an array
        state = state["player_0"]
        MAX_MAP_SIZE = state.env_cfg.map_size.max()
        
        # below maps state.units to across batch dimensions, our team (0), and the first unit (0)
        unit = jax.tree_map(lambda x : x[..., 0, 0], state.units)
        
        # below maps all factory positions to across batch dimensions, our team (0), and the first unit(0)
        factory_pos = jax.tree_map(lambda x : x[..., 0, 0], state.factories.pos.pos) / MAX_MAP_SIZE

        # store cargo+power values scaled to [0, 1]
        cargo_space = state.env_cfg.ROBOTS[1].CARGO_SPACE.max()
        battery_cap = state.env_cfg.ROBOTS[1].BATTERY_CAPACITY.max()
        cargo_vec = jnp.hstack(
            [
                (unit.power / battery_cap)[..., None],
                unit.cargo.stock / cargo_space
            ]
        )
        unit_type = unit.unit_type  # note that build actions use 0 to encode Light
        # normalize the unit position
        pos = unit.pos.pos / MAX_MAP_SIZE
        unit_vec = jnp.concatenate(
            [pos, unit_type[..., None], cargo_vec, unit.team_id[..., None]], axis=-1
        )
        # we add some engineered features down here
        # compute closest ice tile
        ice_map = state.board.ice
        def get_closest_ice_tile(unit, ice_map):
            locs = jnp.argwhere(ice_map, size=64, fill_value=128) / MAX_MAP_SIZE
            pos = unit.pos.pos / MAX_MAP_SIZE
            ice_tile_distances = jnp.mean(
                (locs - pos) ** 2, 1
            )
            closest_ice_tile = locs[jnp.argmin(ice_tile_distances)]
            return closest_ice_tile
        closest_ice_tile = jax.vmap(get_closest_ice_tile)(unit, ice_map) # (B, 2)

        # mask out unit data if there are no units on our team
        obs_mask = jax.tree_map(lambda x : x[..., 0], state.n_units) == 0

        ice_feature = (closest_ice_tile - pos)
        factory_feature = (factory_pos - pos)
        unit_vec = unit_vec.at[obs_mask].set(0)

        obs_vec = jnp.concatenate(
            [unit_vec, factory_feature, ice_feature], axis=-1
        )
        obs_vec = obs_vec.at[obs_mask].set(0)
        return obs_vec # (B, 11)

    # we make this method static so the submission/evaluation code can use this as well
    @staticmethod
    def convert_obs(obs: Dict[str, Any], env_cfg: Any) -> Dict[str, npt.NDArray]:
        observation = dict()
        shared_obs = obs["player_0"]
        ice_map = shared_obs["board"]["ice"]
        ice_tile_locations = np.argwhere(ice_map == 1)

        for agent in obs.keys():
            obs_vec = np.zeros(
                13,
            )

            factories = shared_obs["factories"][agent]
            factory_vec = np.zeros(2)
            for k in factories.keys():
                # here we track a normalized position of the first friendly factory
                factory = factories[k]
                factory_vec = np.array(factory["pos"]) / env_cfg.map_size
                break
            units = shared_obs["units"][agent]
            for k in units.keys():
                unit = units[k]

                # store cargo+power values scaled to [0, 1]
                cargo_space = env_cfg.ROBOTS[unit["unit_type"]].CARGO_SPACE
                battery_cap = env_cfg.ROBOTS[unit["unit_type"]].BATTERY_CAPACITY
                cargo_vec = np.array(
                    [
                        unit["power"] / battery_cap,
                        unit["cargo"]["ice"] / cargo_space,
                        unit["cargo"]["ore"] / cargo_space,
                        unit["cargo"]["water"] / cargo_space,
                        unit["cargo"]["metal"] / cargo_space,
                    ]
                )
                unit_type = (
                    0 if unit["unit_type"] == "LIGHT" else 1
                )  # note that build actions use 0 to encode Light
                # normalize the unit position
                pos = np.array(unit["pos"]) / env_cfg.map_size
                unit_vec = np.concatenate(
                    [pos, [unit_type], cargo_vec, [unit["team_id"]]], axis=-1
                )

                # we add some engineered features down here
                # compute closest ice tile
                ice_tile_distances = np.mean(
                    (ice_tile_locations - np.array(unit["pos"])) ** 2, 1
                )
                # normalize the ice tile location
                closest_ice_tile = (
                    ice_tile_locations[np.argmin(ice_tile_distances)] / env_cfg.map_size
                )
                obs_vec = np.concatenate(
                    [unit_vec, factory_vec - pos, closest_ice_tile - pos], axis=-1
                )
                break
            observation[agent] = obs_vec

        return observation
