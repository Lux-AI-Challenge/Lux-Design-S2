use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Team {
    pub team_id: u64,
    pub faction: String,
    pub water: u64,
    pub metal: u64,
    pub factories_to_place: u64,
    pub factory_strains: Vec<u64>,
    pub place_first: bool,
    pub bid: u64,
}
