import numpy as np
from dataclasses import dataclass

from kits.python.lux.cargo import UnitCargo
@dataclass
class Factory:
    team_id: int
    unit_id: str
    power: int
    cargo: UnitCargo
    pos: np.ndarray
    # lichen tiles connected to this factory
    lichen_tiles: np.ndarray
    env_cfg: dict


    def build_heavy_metal_cost(self):
        unit_cfg = self.env_cfg.ROBOTS["HEAVY"]
        return unit_cfg.METAL_COST
    def build_heavy_power_cost(self):
        unit_cfg = self.env_cfg.ROBOTS["HEAVY"]
        return unit_cfg.POWER_COST
    def can_build_heavy(self):
        return self.power >= self.build_heavy_power_cost() and self.cargo.metal >= self.build_heavy_metal_cost()
    def build_heavy(self):
        return 1

    def build_light_metal_cost(self):
        unit_cfg = self.env_cfg.ROBOTS["HEAVY"]
        return unit_cfg.METAL_COST
    def build_light_power_cost(self):
        unit_cfg = self.env_cfg.ROBOTS["HEAVY"]
        return unit_cfg.POWER_COST
    def can_build_light(self):
        return self.power >= self.build_light_power_cost() and self.cargo.metal >= self.build_light_metal_cost()

    def build_light(self):
        return 0

    def water_power_cost(self, config):
        """
        Power required to perform water action
        """
        return np.ceil(len(self.lichen_tiles) / self.env_cfg.LICHEN_WATERING_COST_FACTOR) + 1
    def can_water(self):
        return self.power >= self.water_cost()
    def water(self):
        return 2