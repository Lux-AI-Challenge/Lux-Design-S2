use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Team {
    team_id: u64,
    faction: String,
    water: u64,
    metal: u64,
    factories_to_place: u64,
    factory_strains: Vec<u64>,
    place_first: bool,
    bid: u64,
}
