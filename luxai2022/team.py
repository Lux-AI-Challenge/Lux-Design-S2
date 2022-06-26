from dataclasses import dataclass
from enum import Enum


@dataclass
class FactionInfo:
    color: str = "none"
    alt_color: str = "red"
    faction_id: int = -1


class FactionTypes(Enum):
    AlphaStrike = FactionInfo(color="yellow", faction_id=0)
    MotherMars = FactionInfo(color="green", faction_id=1)
    TheBuilders = FactionInfo(color="blue", faction_id=2)
    FirstMars = FactionInfo(color="red", faction_id=3)


class Team:
    def __init__(self, team_id: int, faction: FactionTypes = None) -> None:
        self.faction = faction
        self.team_id = team_id
