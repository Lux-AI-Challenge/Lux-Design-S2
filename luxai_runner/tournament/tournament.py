import asyncio
import os
from typing import Dict
from luxai_runner.episode import Episode, EpisodeConfig
from luxai_runner.tournament.config import TournamentConfig
from luxai_runner.tournament.rankingsystem import ELO, Rank
from luxai_runner.tournament import matchmaking
import copy
import os.path as osp
class Player:
    def __init__(self, player_id, file) -> None:
        self.id = player_id
        self.file = file
        self.rank: Rank = None

class Tournament:
    def __init__(self, tournament_config_kwargs = dict(), episode_cfg: EpisodeConfig = None):
        self.global_id = 0
        self.episode_id = 0
        self.cfg = TournamentConfig(**tournament_config_kwargs)
        self.eps_cfg = episode_cfg
        min_agents = min(self.cfg.agents_per_episode)
        assert len(self.cfg.agents) >= min_agents
        assert episode_cfg is not None
        
        # init ranking system
        self.ranking_sys = ELO(K=30,init_rating=1000)

        self.players: Dict[str, Player] = dict()
        for agent in self.cfg.agents:
            self.add_player(agent)
        
        self.match_making_sys = matchmaking.Random([x for x in self.players], self.cfg.agents_per_episode)

    def add_player(self, file, name = None):
        if name is None: name = file
        name = f"{name}_{self.global_id}"
        assert os.path.isfile(file), f"Agent file {file} does not exist"

        self.global_id += 1
        self.players[name] = Player(name, file)
        self.players[name].rank = self.ranking_sys.init_rank_state()
    
    async def run(self):
        episodes = set()
        async def _run_episode_cb(a):
            if a in episodes: episodes.discard(a)
            next_players = self.match_making_sys.next_match()
            eps_cfg = copy.deepcopy(self.eps_cfg)
            save_replay_path_split = osp.splitext(eps_cfg.save_replay_path)
            eps_cfg.save_replay_path = f"{save_replay_path_split[0]}_{self.episode_id}{save_replay_path_split[1]}"
            eps_cfg.players = [self.players[p].file for p in next_players]
            self.episode_id += 1
            
            players = dict()
            for i in range(len(next_players)):
                players[f"player_{i}"] = self.players[next_players[i]]

            task = asyncio.Task(self._run_episode(players, eps_cfg))
            episodes.add(task)
            await task
            await _run_episode_cb(task)
        async def print_results():
            import time
            while True:
                import sys
                import time
                from collections import deque

                lines = []

                lines.append(f"==== {self.cfg.name} ====")
                # lines.append("")
                lines.append(f"{'Player':36.36}| {'Rating':8.8}| {'Episodes':14.14}")
                lines.append("-"*62)
                players_sorted = [self.players[p] for p in self.players]
                players_sorted = sorted(players_sorted, key=lambda p : p.rank.rating, reverse=True)
                for p in players_sorted:
                    rank = p.rank
                    lines.append(f"{p.id:36.36}| {str(rank.rating):8.8}| {str(rank.episodes):14.14}")
                lines.append("-"*62)
                lines.append(f"{len(episodes)} episodes are running")


                for _ in range(len(lines)):
                    sys.stdout.write("\x1b[1A\x1b[2K")
                for i in range(len(lines)):
                    sys.stdout.write(lines[i] + "\n")
                await asyncio.sleep(1)
        await asyncio.gather(_run_episode_cb(None), _run_episode_cb(None), print_results())

    async def _run_episode(self, players: Dict[str, Player], eps_cfg: EpisodeConfig):
        eps = Episode(
            cfg=eps_cfg
        )
        rewards = await eps.run()
        # TODO: there is probably some race condition here, but in long run it's not important
        self.ranking_sys.update(players["player_0"].rank, players["player_1"].rank, rewards["player_0"], rewards["player_1"])
        