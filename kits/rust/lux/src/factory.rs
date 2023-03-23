//! Handles serializing, deserializing and interacting with factory units

use crate::cargo::Cargo;
use crate::config::Config;
use crate::state::State;
use crate::Pos;
use serde::{Deserialize, Serialize};
use std::ops::Range;

/// A representation of a Factory
///
/// A factory is a 3x3 unit that can build units, and grow lichen
///
/// For more information see the [spec](https://www.lux-ai.org/specs-s2#factories)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Factory {
    /// Id discriminating the team Factory belongs to
    ///
    /// valid values are 0 or 1
    pub team_id: u64,

    /// Unique identfier for the factory
    ///
    /// Will be of the form "factory_<unique_index>"
    pub unit_id: String,

    /// Current factory power
    pub power: u64,

    /// Position of c
    pub pos: Pos,

    /// Current materials stored in the factory
    pub cargo: Cargo,

    /// Unique identifier for the lichen strain this factory grows
    pub strain_id: u64,
}

impl Factory {
    /// Gets 3x3 range as two range iterators
    #[inline]
    pub fn occupied_range(&self) -> (Range<i64>, Range<i64>) {
        (
            (self.pos.0 - 1..self.pos.0 + 2),
            (self.pos.1 - 1..self.pos.1 + 2),
        )
    }

    /// Validates if the factory has enough power and metal to build a heavy robot
    #[inline]
    pub fn can_build_heavy(&self, cfg: &Config) -> bool {
        let metal_cost = cfg.robots.heavy.metal_cost;
        let power_cost = cfg.robots.heavy.power_cost;
        self.power >= power_cost && self.cargo.metal >= metal_cost
    }

    /// Validates if the factory has enough power and metal to build a light robot
    #[inline]
    pub fn can_build_light(&self, cfg: &Config) -> bool {
        let metal_cost = cfg.robots.light.metal_cost;
        let power_cost = cfg.robots.light.power_cost;
        self.power >= power_cost && self.cargo.metal >= metal_cost
    }

    /// Gets power cost to water lichen
    ///
    /// # Note
    ///
    /// This associated function is an overestimate of the cost as the
    /// true calculation requires the growth set of tiles rather than the
    /// existing tiles. This matches the python implementation and as such
    /// there are no plans to calculate the real cost from this function
    #[inline]
    pub fn water_cost(&self, state: &State) -> u64 {
        let tile_count = state
            .board
            .tiles
            .iter()
            .filter(|tile| {
                if let Some(lichen) = &tile.lichen {
                    lichen.strain == self.strain_id
                } else {
                    false
                }
            })
            .count() as f64;
        f64::ceil(tile_count / state.env_cfg.lichen_watering_cost_factor) as u64
    }
}
