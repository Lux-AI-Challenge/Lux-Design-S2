use crate::cargo::Cargo;
use serde::{Deserialize, Serialize};
use std::ops::Range;

#[derive(Debug, Serialize, Deserialize)]
pub struct Factory {
    pub team_id: u64,
    pub unit_id: String,
    pub power: u64,
    pub pos: (i64, i64),
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
}
