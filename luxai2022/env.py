from collections import OrderedDict, defaultdict
import functools
from typing import Dict, List, Tuple, Union

import numpy as np
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from luxai2022.actions import Action, FactoryBuildAction, FactoryWaterAction, MoveAction, PickupAction, TransferAction, format_action_vec, format_factory_action, validate_actions, move_deltas

from luxai2022.config import EnvConfig
from luxai2022.factory import Factory
from luxai2022.map.board import Board
from luxai2022.spaces.act_space import get_act_space, get_act_space_init
from luxai2022.spaces.obs_space import get_obs_space
from luxai2022.state import State
from luxai2022.team import FactionTypes, Team
from luxai2022.unit import Unit, UnitType
from luxai2022.utils.utils import is_day


def env():
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env = raw_env()
    # This wrapper is only for environments which print results to the terminal
    env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class LuxAI2022(ParallelEnv):
    metadata = {"render.modes": ["human", "html"], "name": "luxai2022_v0"}

    def __init__(self, max_episode_length=1000):
        # TODO - allow user to override env configs
        default_config = EnvConfig()
        self.env_cfg = default_config
        self.possible_agents = ["player_" + str(r) for r in range(2)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.max_episode_length = max_episode_length

        self.state: State = State(seed_rng=None, seed=-1, env_cfg=self.env_cfg, env_steps=-1, board=None)

        self.seed_rng: np.random.RandomState = None

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        return get_obs_space(config=self.env_cfg, agent=agent)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        if self.env_steps == 0:
            return get_act_space_init(config=self.env_cfg, agent=agent)
        return get_act_space(self.state.units, self.state.factories, config=self.env_cfg, agent=agent)

    def render(self, mode="human"):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        # if len(self.agents) == 2:
        #     string = ("Current state: Agent1: {} , Agent2: {}".format(MOVES[self.state[self.agents[0]]], MOVES[self.state[self.agents[1]]]))
        # else:
        #     string = "Game over"
        print("hello")

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def get_state(self):
        return self.state

    def set_state(self, state: State):
        self.state = state
        self.env_steps = state.env_steps
        self.seed_rng = state.seed_rng
        self.seed = state.seed
        # TODO - throw warning if setting state from a different configuration than initialized with
        self.env_cfg = state.env_cfg

    def reset(self, seed=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.

        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.

        Returns the observations for each agent
        """
        seed_rng = np.random.RandomState(seed=seed)
        self.agents = self.possible_agents[:]
        self.env_steps = 0
        self.seed = seed
        board = Board()
        self.state: State = State(seed_rng=seed_rng, seed=seed, env_cfg=self.state.env_cfg, env_steps=0, board=board)
        for agent in self.possible_agents:
            self.state.units[agent] = OrderedDict()
            self.state.factories[agent] = OrderedDict()
        obs = self.state.get_obs()
        observations = {agent: obs for agent in self.agents}
        return observations

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - dones
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}

        failed_agents = {agent: False for agent in self.agents}
        dones = {agent: False for agent in self.agents}
        
        # Turn 1 logic, handle # TODO Bidding

        if self.env_steps == 0:
            # handle initialization
            for k, a in actions.items():
                if k not in self.agents: raise ValueError(f"Invalid player {k}")
                if "spawns" in a:
                    self.state.teams[k] = Team(team_id=self.agent_name_mapping[k], faction=FactionTypes[a["faction"]])
                    for spawn_loc in a["spawns"]:
                        factory = Factory(self.state.teams[k], unit_id=f"factory_{self.state.global_id}")
                        self.state.global_id += 1
                        self.state.factories[k][factory.unit_id] = factory
                else:
                    # team k loses
                    failed_agents[k] = True

            # TODO return the initial obs, skip all the other parts in this list
        else:
            # 1. Check for malformed actions
            if self.env_cfg.validate_actions: 
                try:
                    for agent, unit_actions in actions.items():
                        if not self.action_space(agent).contains(unit_actions):
                            raise ValueError("Inappropriate action given")
                    for agent, unit_actions in actions.items():
                        for unit_id, action in unit_actions.items():
                            if "factory" in unit_id:
                                self.state.factories[agent][unit_id].action_queue.append(format_factory_action(action))
                            elif "unit" in unit_id:
                                formatted_actions = []
                                if type(action) == list:
                                    formatted_actions = [format_action_vec(a) for a in action]
                                else:
                                    formatted_actions = [format_action_vec(action)]
                                self.state.units[agent][unit_id].action_queue = formatted_actions
                except ValueError:
                    failed_agents[agent] = True

            actions_by_type: Dict[str, List[Tuple[Unit, Action]]] = defaultdict(list)
            for agent in self.agents:
                if failed_agents[agent]: continue
                for unit in self.state.units[agent].values():
                    unit_a = unit.next_action()
                    actions_by_type[unit_a.act_type].append((unit, unit_a))
            
            # 2. validate all actions against current state, throw away impossible actions TODO
            # if self.env_cfg.validate_actions: 
                # actions_by_type = validate_actions(self.env_cfg, self.state, actions_by_type)
            # TODO Transfer resources/power

            for unit, transfer_action in actions_by_type["transfer"]:
                transfer_action: TransferAction
                transfer_amount = unit.sub_resource(transfer_action.resource, transfer_action.transfer_amount)
                transfer_pos = unit.pos + move_deltas[transfer_action.transfer_dir]
                units_there = self.state.board.get_units_at(transfer_pos)
                if units_there is not None:
                    assert len(units_there) == 1
                    target_unit = units_there[0]
                    # add resources to target. This will waste (transfer_amount - actually_transferred) resources
                    actually_transferred = target_unit.add_resource(transfer_action.resource, transfer_amount)
                
            # TODO Resource Pickup
            for unit, pickup_action in actions_by_type["transfer"]:
                pickup_action: PickupAction

            # TODO digging and self destruct

            # TODO execute movement and recharge/wait actions, then resolve collisions
            new_units_map: Dict[str, List[Unit]] = defaultdict(list)
            for unit, move_action in actions_by_type["move"]:
                move_action: MoveAction
                old_pos_hash = self.state.board.pos_hash(unit.pos)
                target_pos = unit.pos + move_action.dist * move_deltas[move_action.move_dir]
                rubble = self.state.board.rubble[target_pos.y, target_pos.x]
                power_required = unit.unit_cfg.MOVE_COST + unit.unit_cfg.RUBBLE_MOVEMENT_COST * rubble
                unit.pos = target_pos
                new_pos_hash = self.state.board.pos_hash(unit.pos)
                del self.state.board.units_map[old_pos_hash]
                new_units_map[new_pos_hash].append(unit)
                unit.power -= power_required
            
            for pos_hash, units in new_units_map:
                if len(units) <= 1: continue
                # TODO handle collisions

            self.state.board.units_map = new_units_map

            # nothing to do for recharging actions

            # TODO - grow lichen

            # TODO - robot building with factories
            for agent in self.agents:
                for factory in self.state.factories[agent].values():
                    if len(factory.action_queue) > 0:
                        action = factory.action_queue.pop()
                        if action.act_type == "factory_build":
                            team = self.state.teams[agent]
                            unit = Unit(team=team, unit_type=UnitType.HEAVY if action.unit_type == 1 else UnitType.LIGHT, unit_id=f"unit_{self.state.global_id}", env_cfg=self.env_cfg)
                            unit.pos.pos = factory.pos.pos.copy()
                            if unit.unit_type == UnitType.HEAVY:
                                # TODO allow user to specify how much power to give unit?
                                unit.power = 500
                            else:
                                unit.power = 50
                            self.state.global_id += 1
                            self.state.units[agent][unit.unit_id] = unit

            # resources refining
            for agent in self.agents:
                for factory in self.state.factories[agent].values():
                    factory.refine_step(self.env_cfg)
            # power gain
            if is_day(self.env_cfg, self.env_steps):
                for agent in self.agents:
                    for u in self.state.units[agent].values():
                        u.power = u.power + self.env_cfg.ROBOTS[u.unit_type].CHARGE
            for agent in self.agents:
                for f in self.state.factories[agent].values():
                    f.power = f.power + self.env_cfg.FACTORY_CHARGE    

        # rewards for all agents are placed in the rewards dictionary to be returned
        rewards = {}
        for agent in self.agents:
            unit_ids = list(self.state.factories[agent].keys())
            # TODO: TEST
            rewards[agent] = self.state.board.lichen[np.isin(self.state.board.lichen_strains, unit_ids)].sum()

        self.env_steps += 1
        env_done = self.env_steps >= self.max_episode_length
        

        # generate observations
        obs = self.state.get_obs()
        observations = {}
        for k in self.agents:
            observations[k] = obs

        # log stats and other things
        infos = {agent: {} for agent in self.agents}

        if env_done:
            self.agents = []

        return observations, rewards, dones, infos

    ### Game Logic ###
    def add_unit(self, team_id):
        u = Unit(team=Team(1, FactionTypes.MotherMars), unit_type=UnitType.HEAVY, unit_id="1s")


def raw_env() -> LuxAI2022:
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = LuxAI2022()
    # env = parallel_to_aec(env)
    return env


if __name__ == "__main__":
    env: LuxAI2022 = LuxAI2022()
    o = env.reset()
    # u = Unit(team=Team(1, FactionTypes.MotherMars), unit_type=UnitType.HEAVY, unit_id='1s')
    # env.state.units[1].append(u)
    # observation, reward, done, info = env.last()
    o, r, d, _ = env.step(
        {"player_0": dict(faction="MotherMars", spawns=np.array([[4, 4], [15, 5]])), "player_1": dict(faction="AlphaStrike", spawns=np.array([[56, 55], [40, 42]]))}
    )
    print(o, r, d)

    all_actions = dict()
    for team_id, agent in enumerate(env.possible_agents):
        obs = o[agent]
        all_actions[agent] = dict()
        # units = o[agent]["units"]
        # actions = []
        # for unit_id, unit in units.items():
        #     actions.append(dict(unit_id=unit_id))
        factories = obs["factories"][agent]
        actions = dict()
        for unit_id, factory in factories.items():
            actions[unit_id] = 0
        all_actions[agent] = actions

    o, r, d, _ = env.step(all_actions)
    import ipdb

    ipdb.set_trace()
