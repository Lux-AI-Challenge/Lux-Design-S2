//! Unit cargo

use serde::{Deserialize, Serialize};

/// Represents the amount of each resource owned by a unit
#[allow(missing_docs)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cargo {
    pub ice: u64,
    pub ore: u64,
    pub water: u64,
    pub metal: u64,
}
