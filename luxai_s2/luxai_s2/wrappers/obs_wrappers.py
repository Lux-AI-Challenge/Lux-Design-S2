from typing import Dict

import gym
import numpy as np
import numpy.typing as npt
from gym import spaces

import luxai_s2.env
from luxai_s2.env import LuxAI_S2
from luxai_s2.state import ObservationStateDict
class SimpleUnitObservationWrapper(gym.ObservationWrapper):
    """
    A simple state based observation to work with in pair with the SimpleUnitDiscreteController

    It contains info only on the first robot, the first factory you own, and some useful features. If there are no owned robots the observation is just zero.
    No information about the opponent is included

    Included features:
    - First robot's stats
    - distance vector to closest ice tile
    - distance vector to first factory

    """

    def __init__(self, env: gym.Env, max_robots: int = 5) -> None:
        super().__init__(env)
        self.max_robots = max_robots
        self.observation_space = spaces.Box(-999, 999, shape=(13 * max_robots,))

    def observation(
        self, obs: Dict[str, ObservationStateDict]
    ) -> Dict[str, npt.NDArray]:
        observation = dict()
        shared_obs = obs["player_0"]
        ice_map = shared_obs["board"]["ice"]
        ice_tile_locations = np.argwhere(ice_map == 1)
        unit_ct = 0
        for agent in obs.keys():
            factories = shared_obs["factories"][agent]
            factory_vec = np.zeros(2)
            for k in factories.keys():
                # here we track a normalized position of the first friendly factory
                factory = factories[k]
                factory_vec = np.array(factory["pos"]) / self.env.state.env_cfg.map_size
                break
            units = shared_obs["units"][agent]
            unit_ct = 0
            unit_obs_vecs = []
            for k in units.keys():
                unit = units[k]

                # store cargo+power values scaled to [0, 1]
                cargo_space = self.env.state.env_cfg.ROBOTS[
                    unit["unit_type"]
                ].CARGO_SPACE
                battery_cap = self.env.state.env_cfg.ROBOTS[
                    unit["unit_type"]
                ].BATTERY_CAPACITY
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
                pos = np.array(unit["pos"]) / self.env.state.env_cfg.map_size
                unit_vec = np.concatenate(
                    [pos, [unit_type], cargo_vec, [unit["team_id"]]], axis=-1
                )
                # engineered features

                # compute closest ice tile
                ice_tile_distances = np.mean(
                    (ice_tile_locations - np.array(unit["pos"])) ** 2, 1
                )
                # normalize the ice tile location
                closest_ice_tile = (
                    ice_tile_locations[np.argmin(ice_tile_distances)]
                    / self.env.state.env_cfg.map_size
                )
                obs_vec = np.concatenate(
                    [unit_vec, factory_vec - pos, closest_ice_tile - pos], axis=-1
                )
                unit_obs_vecs += [obs_vec]
                unit_ct += 1
                if unit_ct >= self.max_robots:
                    break
            # pad with zero vectors
            if self.max_robots - unit_ct > 0:
                unit_obs_vecs.append(np.zeros(13 * (self.max_robots - unit_ct)))
            observation[agent] = np.concatenate(unit_obs_vecs, axis=-1)

        return observation