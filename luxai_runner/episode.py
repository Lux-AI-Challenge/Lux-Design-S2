
import asyncio
from dataclasses import dataclass
import time
from typing import Any, Callable, Dict, List, Optional
import gym
from luxai_runner.logger import Logger
from luxai_runner.bot import Bot
import numpy as np

from luxai_runner.utils import to_json
@dataclass
class EpisodeConfig:
    players: List[str]
    env_cls: Callable[[Any], gym.Env]
    seed: Optional[int] = None
    env_cfg: Optional[Any] = dict
    verbosity: Optional[int] = 1
    render: Optional[bool] = True

class Episode:
    def __init__(self, cfg: EpisodeConfig) -> None:
        self.cfg = cfg
        self.env = cfg.env_cls(**cfg.env_cfg)
        self.log = Logger(identifier="Episode", verbosity=cfg.verbosity)
        self.seed = cfg.seed if cfg.seed is not None else np.random.randint(9999999)
        self.players = cfg.players

    async def run(self):
        if len(self.players) != 2: 
            raise ValueError("Must provide two paths.")
        # Start agents
        players: Dict[str, Bot] = dict()
        start_tasks = []
        for i in range(2):
            player = Bot(self.players[i], f"player_{i}", i, verbose=self.log.verbosity)
            player.proc.log.identifier = player.log.identifier
            players[player.agent] = player
            start_tasks += [player.proc.start()]
        await asyncio.wait(start_tasks, return_when=asyncio.ALL_COMPLETED)

        
        obs = self.env.reset(seed=self.seed)
        state_obs = self.env.state.get_compressed_obs()
        obs = to_json(state_obs)

        if self.cfg.render: 
            self.env.render()
            time.sleep(0.2)
        game_done = False
        rewards, dones, infos = dict(), dict(), dict()
        for agent in players:
            rewards[agent] = 0 
            dones[agent] = 0
            infos[agent] = dict()
        i= 0
        while not game_done:
            i += 1
            # print("===", self.env.env_steps)
            actions = dict()
            
            agent_ids = []
            action_coros = []
            for player in players.values():
                action = player.step(obs, self.env.env_steps, rewards[agent], infos[agent])
                action_coros += [action]
                agent_ids += [player.agent]
            resolved_actions = await asyncio.gather(*action_coros)
            for agent_id, action in zip(agent_ids, resolved_actions):
                actions[agent_id] = action
            new_state_obs, rewards, dones, infos = self.env.step(actions)
            obs = self.env.state.get_change_obs(state_obs)
            state_obs = new_state_obs["player_0"]
            obs = to_json(obs)

            if self.cfg.render: 
                self.env.render()
                time.sleep(0.2)
            players_left = len(dones)
            for k in dones:
                if dones[k]: players_left -= 1
            if players_left < 2: # specific to lux ai 2022
                game_done = True
        print("Final Scores", rewards)