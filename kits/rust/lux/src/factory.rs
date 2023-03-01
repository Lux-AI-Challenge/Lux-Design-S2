use crate::cargo::Cargo;
use crate::config::Config;
use crate::state::State;
use crate::Pos;
use serde::{Deserialize, Serialize};
use std::ops::Range;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Factory {
    pub team_id: u64,
    pub unit_id: String,
    pub power: u64,
    pub pos: Pos,
    pub cargo: Cargo,
    pub strain_id: u64,
}

impl Factory {
    #[inline]
    pub fn occupied_range(&self) -> (Range<i64>, Range<i64>) {
        // TODO(seamooo) is this guaranteed to be in bounds?
        (
            (self.pos.0 - 1..self.pos.0 + 2),
            (self.pos.1 - 1..self.pos.1 + 2),
        )
    }
    #[inline]
    pub fn can_build_heavy(&self, cfg: &Config) -> bool {
        let metal_cost = cfg.robots.heavy.metal_cost;
        let power_cost = cfg.robots.heavy.power_cost;
        self.power >= power_cost && self.cargo.metal >= metal_cost
    }
    #[inline]
    pub fn can_build_light(&self, cfg: &Config) -> bool {
        let metal_cost = cfg.robots.light.metal_cost;
        let power_cost = cfg.robots.light.power_cost;
        self.power >= power_cost && self.cargo.metal >= metal_cost
    }
    #[inline]
    pub fn water_cost(&self, state: &State) -> u64 {
        let tile_count = state
            .board
            .lichen_strains
            .iter()
            .flat_map(|x| x.iter())
            .filter(|strain_id| **strain_id == self.strain_id)
            .count() as f64;
        f64::ceil(tile_count / state.env_cfg.lichen_watering_cost_factor) as u64
    }
}
