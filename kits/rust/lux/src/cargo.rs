use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Cargo {
    ice: u64,
    ore: u64,
    water: u64,
    metal: u64,
}
