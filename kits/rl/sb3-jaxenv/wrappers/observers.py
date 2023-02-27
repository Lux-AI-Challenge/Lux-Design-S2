from abc import ABC, abstractmethod
from gym import spaces
from jux.env import State as JuxState
from functools import partial
from jux.actions import FactoryAction, UnitAction, UnitActionType
from jux.env import JuxAction, JuxEnv, JuxEnvBatch
from jux.env import State as JuxState
from jux.unit import UnitType, Unit
import jax
import jax.numpy as jnp
class Observer(ABC):
    def __init__(self, observation_space: spaces.Space) -> None:
        self.observation_space = observation_space
    @abstractmethod
    def convert_jux_obs(self, state: JuxState, agent: int):
        """
        Given a state/observation (JuxState) for this agent (int), return 
        a observation conforming to the defined observation space\
        
        Should defined so that it can be jitted. If you write your own Observer, make sure to add
        `@partial(jax.jit, static_argnums=(0,))`
        """
        raise NotImplementedError()
class SimpleUnitObserver(Observer):
    """
    A simple state based observation to work with in pair with the SimpleUnitDiscreteController

    It contains info only on the first robot, the first factory you own, and some useful features. If there are no owned robots the observation is just zero.
    No information about the opponent is included. This will generate observations for all teams.

    Included features:
    - First robot's stats
    - distance vector to closest ice tile
    - distance vector to first factory

    """
    def __init__(self) -> None:
        observation_space = spaces.Box(-999, 999, shape=(11,))
        super().__init__(observation_space)

    @partial(jax.jit, static_argnums=(0,))
    def convert_jux_obs(self, state: JuxState, agent: int):
        # converts a single state to a single observation

        # we know that the jax state of lux always has units and factories at the front of an array
        MAX_MAP_SIZE = state.env_cfg.map_size.max()

        # index by team (agent), and the first unit (0) to get first unit of team agent
        # the jax.tree_map tool can make it simple to index across all leaves in an nested object like state or state.units
        unit: Unit = jax.tree_map(lambda x: x[agent, 0], state.units)

        # below maps all factory positions to across batch dimensions, our team (0), and the first unit(0)
        factory_pos = (
            jax.tree_map(lambda x: x[agent, 0], state.factories.pos.pos) / MAX_MAP_SIZE
        )

        # store cargo+power values scaled to [0, 1]
        cargo_space = state.env_cfg.ROBOTS[1].CARGO_SPACE.max()
        battery_cap = state.env_cfg.ROBOTS[1].BATTERY_CAPACITY.max()
        cargo_vec = jnp.hstack(
            [(unit.power / battery_cap)[..., None], unit.cargo.stock / cargo_space]
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

        # def get_closest_ice_tile(unit, ice_map):
        locs = jnp.argwhere(ice_map, size=64, fill_value=128) / MAX_MAP_SIZE
        pos = unit.pos.pos / MAX_MAP_SIZE
        ice_tile_distances = jnp.mean((locs - pos) ** 2, 1)
        closest_ice_tile = locs[jnp.argmin(ice_tile_distances)]
            # return closest_ice_tile

        # closest_ice_tile = jax.vmap(get_closest_ice_tile)(unit, ice_map)  # (B, 2)

        # mask out unit data if there are no units on our team
        own_units = state.n_units[agent] != 0

        ice_feature = closest_ice_tile - pos
        factory_feature = factory_pos - pos

        obs_vec = jnp.concatenate([unit_vec, factory_feature, ice_feature], axis=-1)
        obs_vec = obs_vec * own_units
        return obs_vec  # (11, )
