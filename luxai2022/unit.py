from enum import Enum

import numpy as np
from termcolor import colored

from luxai2022.globals import TERM_COLORS
from luxai2022.map.position import Position
from luxai2022.team import FactionTypes, Team


class UnitType(Enum):
    LIGHT = "Light"
    HEAVY = "Heavy"
    FACTORY = "Factory"
    


class Unit:
    def __init__(self, team: Team, unit_type: str, unit_id: str) -> None:
        self.unit_type = unit_type
        self.team_id = team.team_id
        self.team = team
        self.unit_id = unit_id
        self.pos = Position(np.zeros(2, dtype=int))

    def __str__(self) -> str:
        out = f"[{self.team_id}] {self.unit_type} at {self.pos}"
        if TERM_COLORS:
            return colored(out, self.team.faction.value.color)
        return out
    def is_mobile(self) -> bool:
        return self.unit_type.value != UnitType.FACTORY
    # def move(self, ) -> str:
    #     if self.is_mobile:
    #     else:
    #         raise TypeError("Unit is not a mobile unit")


if __name__ == "__main__":
    u = Unit(team=Team(0, FactionTypes.AlphaStrike), unit_type=UnitType.HEAVY)
    print(u)
    u = Unit(team=Team(1, FactionTypes.MotherMars), unit_type=UnitType.HEAVY)
    u.pos += np.array([1, 4])
    print(u)
    # u = Unit(team=Team(0, FactionTypes.AlphaStrike), unit_type=UnitType.HEAVY)
    # print(u)
