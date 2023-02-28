use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Faction {
    AlphaStrike,
    MotherMars,
    TheBuilders,
    FirstMars,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Team {
    pub team_id: u64,
    pub faction: Faction,
    pub water: u64,
    pub metal: u64,
    pub factories_to_place: u64,
    pub factory_strains: Vec<u64>,
    pub place_first: bool,
    pub bid: u64,
}
