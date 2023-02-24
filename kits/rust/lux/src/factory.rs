use crate::cargo::Cargo;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Factory {
    team_id: u64,
    unit_id: String,
    power: u64,
    pos: [u64; 2],
    cargo: Cargo,
    strain_id: u64,
}
