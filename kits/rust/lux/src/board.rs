use crate::utils::OpaqueRectArrDbg;
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Serialize, Deserialize)]
pub struct Board {
    pub rubble: Vec<Vec<u64>>,
    pub ore: Vec<Vec<u64>>,
    pub lichen: Vec<Vec<u64>>,
    pub lichen_strains: Vec<Vec<u64>>,
    pub valid_spawns_mask: Vec<Vec<bool>>,
    pub factory_occupancy: Vec<Vec<u64>>,
    pub factories_per_team: u64,
}

impl fmt::Debug for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        let rubble = OpaqueRectArrDbg(&self.rubble);
        let ore = OpaqueRectArrDbg(&self.ore);
        let lichen = OpaqueRectArrDbg(&self.lichen);
        let lichen_strains = OpaqueRectArrDbg(&self.lichen_strains);
        let valid_spawns_mask = OpaqueRectArrDbg(&self.valid_spawns_mask);
        let factory_occupancy = OpaqueRectArrDbg(&self.factory_occupancy);
        f.debug_struct("Board")
            .field("rubble", &rubble)
            .field("ore", &ore)
            .field("lichen", &lichen)
            .field("lichen_strains", &lichen_strains)
            .field("valid_spawns_mask", &valid_spawns_mask)
            .field("factory_occupancy", &factory_occupancy)
            .field("factories_per_team", &self.factories_per_team)
            .finish()
    }
}
