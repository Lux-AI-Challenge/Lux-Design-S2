from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TournamentConfig:
    name: str = "luxai_s2_tourney"
    tournament_type: str = "leadeboard"  # roundrobin, leaderboard
    ranking_system: str = "elo"  # elo, win/loss, TODO: Bradley-Terry
    ranking_system_cfg: Dict = field(default_factory=dict)
    matchmaking_system: str = "random"

    agents: List[str] = field(default_factory=list)

    agents_per_episode: List[int] = field(default_factory=lambda: [2])

    max_concurrent_episodes: int = 4
