from abc import ABC
from typing import List

import numpy as np


class MatchMakingSystem(ABC):
    def __init__(
        self, agents: List[str], agents_per_episode: List[int], seed=0
    ) -> None:
        super().__init__()
        self.agents = agents
        self.agents_per_episode = agents_per_episode
        self.np_random = np.random.RandomState(seed=seed)

    def next_match(self) -> List[str]:
        pass


class Random(MatchMakingSystem):
    def __init__(
        self, agents: List[str], agents_per_episode: List[int], seed=0
    ) -> None:
        super().__init__(agents, agents_per_episode, seed)

    def next_match(self) -> List[str]:
        num_agents = self.np_random.choice(self.agents_per_episode)
        agents = self.np_random.choice(self.agents, size=num_agents, replace=False)
        return agents
