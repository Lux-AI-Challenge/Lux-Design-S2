
import asyncio
from dataclasses import dataclass
import json
import time
from typing import Any, Callable, Dict, List, Optional
import gym
from luxai_runner.logger import Logger
from luxai_runner.bot import Bot
import numpy as np
import dataclasses
from luxai_runner.utils import to_json
import os.path as osp

@dataclass 
class ReplayConfig:
    save_format: str = "json"
    compressed_obs: bool = False
@dataclass
class EpisodeConfig:
    players: List[str]
    env_cls: Callable[[Any], gym.Env]
    seed: Optional[int] = None
    env_cfg: Optional[Any] = dict
    verbosity: Optional[int] = 1
    render: Optional[bool] = True
    save_replay_path: Optional[str] = None
    replay_options: ReplayConfig = ReplayConfig()

class Episode:
    def __init__(self, cfg: EpisodeConfig) -> None:
        self.cfg = cfg
        self.env = cfg.env_cls(**cfg.env_cfg)
        self.log = Logger(identifier="Episode", verbosity=cfg.verbosity)
        self.seed = cfg.seed if cfg.seed is not None else np.random.randint(9999999)
        self.players = cfg.players

    def save_replay(self, replay):
        # JSON option
        if self.cfg.replay_options.save_format == "json":
            replay["observations"] = [to_json(x) for x in replay["observations"]]
            replay["actions"] = [to_json(x) for x in replay["actions"]]
            del replay["dones"]
            del replay["rewards"]
            ext = ".json"
            from pathlib import Path
            dir_name = osp.dirname(self.cfg.save_replay_path)
            if dir_name != "":
                Path(dir_name).mkdir(parents=True, exist_ok=True)
            if self.cfg.save_replay_path[-5:] == ".json":
                ext = ""
            with open(f"{self.cfg.save_replay_path}{ext}", "w") as f:
                json.dump(replay, f)
        else:
            raise ValueError(f"{self.cfg.replay_options.save_format} is not a valid save format")

    async def run(self):
        if len(self.players) != 2: 
            raise ValueError("Must provide two paths.")
        # Start agents
        players: Dict[str, Bot] = dict()
        start_tasks = []
        save_replay = self.cfg.save_replay_path is not None
        for i in range(2):
            player = Bot(self.players[i], f"player_{i}", i, verbose=self.log.verbosity)
            player.proc.log.identifier = player.log.identifier
            players[player.agent] = player
            start_tasks += [player.proc.start()]
        await asyncio.wait(start_tasks, return_when=asyncio.ALL_COMPLETED)

        
        obs = self.env.reset(seed=self.seed)
        env_cfg = self.env.state.env_cfg
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
            infos[agent] = dict(
                # turn 0 provide configurations
                env_cfg=dataclasses.asdict(env_cfg)
            )

        if save_replay:
            replay = dict(observations=[], actions=[], dones=[], rewards=[])
            if self.cfg.replay_options.compressed_obs:
                replay["observations"].append(state_obs)
            else:
                replay["observations"].append(self.env.state.get_obs())

        i = 0
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
                try:
                    for k in action:
                        if type(action[k]) == list:
                            action[k] = np.array(action[k])
                    actions[agent_id] = action
                except:
                    if self.cfg.verbosity > 0:
                        if action is None:
                            print(f"{agent_id} sent a invalid action {action}. Agent likely errored out somewhere, check above for stderr logs")
                        else:
                            print(f"{agent_id} sent a invalid action {action}")
                    actions[agent_id] = None
            new_state_obs, rewards, dones, infos = self.env.step(actions)
            change_obs = self.env.state.get_change_obs(state_obs)
            state_obs = new_state_obs["player_0"]
            obs = to_json(change_obs)
            if save_replay:
                if self.cfg.replay_options.compressed_obs:
                    replay["observations"].append(change_obs)
                else:
                    replay["observations"].append(self.env.state.get_obs())
                replay["actions"].append(actions)
                replay["rewards"].append(rewards)
                replay["dones"].append(dones)

            if self.cfg.render: 
                self.env.render()
                time.sleep(0.1)
            players_left = len(dones)
            for k in dones:
                if dones[k]: players_left -= 1
            if players_left < 2: # specific to lux ai 2022
                game_done = True
        self.log.info(f"Final Scores: {rewards}")
        if save_replay:
            self.save_replay(replay)

        for player in players.values():
            await player.proc.cleanup()

        return rewards
