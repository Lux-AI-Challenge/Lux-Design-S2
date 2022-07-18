import numpy as np
from luxai2022.map_generator.generator import GameMap, random_map


class Board:
    def __init__(self) -> None:
        self.map: GameMap = random_map()
        self.height = self.map.height
        self.width = self.map.width
        self.units_map: np.ndarray = np.zeros((self.width, self.height))

        

