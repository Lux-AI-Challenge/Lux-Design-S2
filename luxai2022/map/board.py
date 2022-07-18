from collections import defaultdict
from typing import Dict, List
import numpy as np
from luxai2022.map.position import Position
from luxai2022.map_generator.generator import GameMap
from luxai2022.unit import Unit


class Board:
    def __init__(self) -> None:
        self.height = 64
        self.width = 64
        # self.map: GameMap = random_map()
        map_type = None #args.get("map_type", None)
        symmetry = None # args.get("symmetry", None)
        # TODO fix rng here
        self.map = GameMap.random_map(seed=0, symmetry=symmetry, map_type=map_type, width=self.width, height=self.height)
        self.lichen = np.zeros((self.height, self.width))
        self.lichen_strains = np.zeros((self.height, self.width)) # ownership of lichen
        # self.units_map: np.ndarray = np.zeros((self.height, self.width))
        self.units_map: Dict[str, List[Unit]] = defaultdict(list)
    def pos_hash(self, pos: Position):
        return f"{pos.x},{pos.y}"
    def get_units_at(self, pos: Position):
        pos_hash = f"{pos.x},{pos.y}"
        if pos_hash in self.units_map:
            return self.units_map[pos_hash]
        return None
    @property
    def rubble(self):
        return self.map.rubble
    @property
    def ice(self):
        return self.map.ice
    @property
    def ore(self):
        return self.map.ore
    def state_dict(self):
        return dict(
            rubble=self.rubble,
            ore=self.ore,
            ice=self.ice,
            lichen=self.lichen,
            lichen_strains=self.lichen_strains

        )
        

        
        
        

