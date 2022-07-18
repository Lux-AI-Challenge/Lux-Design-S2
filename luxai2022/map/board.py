import numpy as np
from luxai2022.map_generator.generator import GameMap


class Board:
    def __init__(self) -> None:
        self.height = 64
        self.width = 64
        # self.map: GameMap = random_map()
        map_type = None #args.get("map_type", None)
        symmetry = None # args.get("symmetry", None)
        # TODO fix rng here
        self.map = GameMap.random_map(seed=0, symmetry=symmetry, map_type=map_type, width=self.width, height=self.height)
        
        self.units_map: np.ndarray = np.zeros((self.width, self.height))
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
            ice=self.ice
        )
        

        
        
        

