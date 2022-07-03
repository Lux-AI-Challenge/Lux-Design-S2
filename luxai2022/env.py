import functools
from typing import Dict, List

import numpy as np
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers

from luxai2022.config import EnvConfig
from luxai2022.spaces.act_space import get_act_space, get_act_space_init
from luxai2022.spaces.obs_space import get_obs_space
from luxai2022.state import State
from luxai2022.team import FactionTypes, Team
from luxai2022.unit import Unit, UnitType


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
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.max_episode_length = max_episode_length

        self.state: State = State(seed_rng=None, seed=-1, env_cfg=self.env_cfg, env_steps=-1)

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
        return get_act_space(self.state.units, config=self.env_cfg, agent=agent)

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
        self.state: State = State(seed_rng=seed_rng, seed=seed, env_cfg=self.state.env_cfg, env_steps=0)
        for agent in range(len(self.possible_agents)):
            self.state.units[agent] = []
        observations = {agent: 0 for agent in self.agents}
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

        # TODO - format actions

        # Turn 1 logic, handle # TODO Bidding
        
        if self.env_steps == 0:
            # handle initialization
            for k, a in actions.items():
                print(k, a, self.state.teams)
                self.state.teams[k] = Team(team_id=self.agent_name_mapping[k], faction=a["faction"])

        # rewards for all agents are placed in the rewards dictionary to be returned
        rewards = {}
        rewards[self.agents[0]], rewards[self.agents[1]] = (0, 0)

        self.env_steps += 1
        env_done = self.env_steps >= self.max_episode_length
        dones = {agent: env_done for agent in self.agents}

        # current observation is just the other player's most recent action
        observations = {}
        for k in self.agents:
            observations[k] = 0

        # log stats and other things
        infos = {agent: {} for agent in self.agents}

        if env_done:
            self.agents = []

        return observations, rewards, dones, infos

    ### Game Logic ###
    def add_unit(self):
        u = Unit(team=Team(1, FactionTypes.MotherMars), unit_type=UnitType.HEAVY, unit_id='1s')


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
    u = Unit(team=Team(1, FactionTypes.MotherMars), unit_type=UnitType.HEAVY, unit_id='1s')
    env.state.units[1].append(u)
    # observation, reward, done, info = env.last()
    print("obs", o)
    o, r, d, _ = env.step({"player_0": dict(faction="MotherMars"), "player_1": dict(faction="AlphaStrike")})
    print(o, r, d)
    import ipdb;ipdb.set_trace()