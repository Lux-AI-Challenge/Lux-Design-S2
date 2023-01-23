from typing import Dict, Callable

import gym
import numpy as np
from gym import spaces
import numpy.typing as npt
import luxai_s2.env
from luxai_s2.env import LuxAI_S2
from luxai_s2.state import ObservationStateDict
from luxai_s2.unit import ActionType, FactoryPlacementActionType, BidActionType
from luxai_s2.utils import my_turn_to_place_factory
from luxai_s2.wrappers.controllers import Controller, SimpleDiscreteController, SimpleSingleUnitDiscreteController


class SB3Wrapper(gym.Wrapper):
    def __init__(
        self,
        env: LuxAI_S2,
        bid_policy: Callable[[str, ObservationStateDict], Dict[str, BidActionType]]=None,
        factory_placement_policy: Callable[[str, ObservationStateDict], Dict[str, FactoryPlacementActionType]]=None,
        heuristic_policy: Callable[[str, ObservationStateDict], Dict[str, ActionType]]=None,
        controller: Controller=None,
    ) -> None:
        """
        A environment wrapper for Stable Baselines 3 that simplifies the observation and action space. It also reduces the LuxAI_S2 env
        into a single phase game and places the first two phases (bidding and factory placement) into the env.reset function so that
        interacting agents directly start generating actions to play the third phase of the game.

        Parameters
        ----------
        bid_policy: Function
            A function accepting player: str and obs: ObservationStateDict as input that returns a bid action
            such as dict(bid=10, faction="AlphaStrike"). By default will bid 0
        factory_placement_policy: Function
            A function accepting player: str and obs: ObservationStateDict as input that returns a factory placement action
            such as dict(spawn=np.array([2, 4]), metal=150, water=150). By default will spawn in a random valid location with metal=150, water=150
        controller : Controller
            A controller that parameterizes the action space into something more usable and converts parameterized actions to lux actions.
            See luxai_s2/wrappers/controllers.py for available controllers and how to make your own
        
        heuristic_policy: Function
            A function accepting player: str and obs: ObservationStateDict as input and returns a lux action. This can be provided by the user
            to define custom logic or a model to generate actions for any of the units or factories. For any action generate for a unit or factory, it will
            override the original action for that unit or factory when the step function is called. By defalt this is None and not used
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

            def factory_placement_policy(player, obs: ObservationStateDict):
                potential_spawns = np.array(
                    list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1)))
                )
                spawn_loc = potential_spawns[
                    np.random.randint(0, len(potential_spawns))
                ]
                return dict(spawn=spawn_loc, metal=150, water=150)

        self.factory_placement_policy = factory_placement_policy
        if bid_policy is None:

            def bid_policy(player, obs: ObservationStateDict):
                faction = "AlphaStrike"
                if player == "player_1":
                    faction = "MotherMars"
                return dict(bid=0, faction=faction)

        self.bid_policy = bid_policy

        self.heuristic_policy = heuristic_policy

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
            if self.heuristic_policy is not None:
                heuristic_lux_action = self.heuristic_policy(agent, self._prev_obs[agent])
                # override keys
                for k in heuristic_lux_action:
                    lux_action[agent][k] = heuristic_lux_action[k]
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
                cargo_space = self.env.state.env_cfg.ROBOTS[unit["unit_type"]].CARGO_SPACE
                battery_cap = self.env.state.env_cfg.ROBOTS[unit["unit_type"]].BATTERY_CAPACITY
                cargo_vec = np.array([unit["power"] / battery_cap, unit["cargo"]["ice"]/cargo_space, unit["cargo"]["ore"]/cargo_space, unit["cargo"]["water"]/cargo_space, unit["cargo"]["metal"]/cargo_space])
                unit_type = 0 if unit["unit_type"] == "LIGHT" else 1 # note that build actions use 0 to encode Light
                unit_vec = np.concatenate([unit["pos"], [unit_type], cargo_vec, [unit["team_id"]]], axis=-1)
                unit_vec[:2] /= self.env.state.env_cfg.map_size

                # note that all data is stored as map[x, y] format
                unit_data[unit["pos"][0], unit["pos"][1]] = unit_vec
                unit_mask[unit["pos"][0], unit["pos"][1]] = 1

            for unit_id in factories.keys():
                factory = factories[unit_id]
                # we encode everything but strain_id or unit_id
                cargo_vec = np.array([factory["power"], factory["cargo"]["ice"], factory["cargo"]["ore"], factory["cargo"]["water"], factory["cargo"]["metal"]])
                cargo_vec = cargo_vec * 1/1000

                factory_vec = np.concatenate([factory["pos"], cargo_vec, [factory["team_id"]]], axis=-1)
                factory_vec[:2] /= self.env.state.env_cfg.map_size
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

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
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