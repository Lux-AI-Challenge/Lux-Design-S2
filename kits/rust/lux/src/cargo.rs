use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cargo {
    pub ice: u64,
    pub ore: u64,
    pub water: u64,
    pub metal: u64,
}
