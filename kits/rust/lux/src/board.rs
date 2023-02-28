use crate::factory::Factory;
use crate::utils::OpaqueRectArrDbg;
use crate::Pos;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

#[derive(Deserialize)]
pub(crate) struct BoardData {
    rubble: Vec<Vec<u64>>,
    ore: Vec<Vec<u64>>,
    lichen: Vec<Vec<u64>>,
    lichen_strains: Vec<Vec<u64>>,
    valid_spawns_mask: Vec<Vec<bool>>,
    factories_per_team: u64,
}

#[derive(Serialize)]
pub(crate) struct BoardDataRef<'a> {
    rubble: &'a Vec<Vec<u64>>,
    ore: &'a Vec<Vec<u64>>,
    lichen: &'a Vec<Vec<u64>>,
    lichen_strains: &'a Vec<Vec<u64>>,
    valid_spawns_mask: &'a Vec<Vec<bool>>,
    factories_per_team: &'a u64,
}

impl<'a> From<&'a Board> for BoardDataRef<'a> {
    fn from(val: &'a Board) -> Self {
        Self {
            rubble: &val.rubble,
            ore: &val.ore,
            lichen: &val.lichen,
            lichen_strains: &val.lichen_strains,
            valid_spawns_mask: &val.valid_spawns_mask,
            factories_per_team: &val.factories_per_team,
        }
    }
}

#[derive(Clone)]
pub struct Board {
    pub rubble: Vec<Vec<u64>>,
    pub ore: Vec<Vec<u64>>,
    pub lichen: Vec<Vec<u64>>,
    pub lichen_strains: Vec<Vec<u64>>,
    pub valid_spawns_mask: Vec<Vec<bool>>,
    pub factory_occupancy: Vec<Vec<Option<u64>>>,
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

impl Board {
    pub(crate) fn from_data_and_factories(
        board_data: BoardData,
        factories: &HashMap<String, HashMap<String, Factory>>,
    ) -> Self {
        let m = board_data.rubble.len();
        let n = if board_data.rubble.is_empty() {
            0
        } else {
            board_data.rubble[0].len()
        };
        // below breaks if there are overlapping factories
        let factory_occupancy = {
            let mut rv: Vec<Vec<Option<u64>>> =
                (0..m).map(|_| (0..n).map(|_| None).collect()).collect();
            for factory_infos in factories.values() {
                for factory in factory_infos.values() {
                    let (x_range, y_range) = factory.occupied_range();
                    for i in x_range {
                        for j in y_range.clone() {
                            rv[i as usize][j as usize] = Some(factory.strain_id);
                        }
                    }
                }
            }
            rv
        };
        Self {
            rubble: board_data.rubble,
            ore: board_data.ore,
            lichen: board_data.lichen,
            lichen_strains: board_data.lichen_strains,
            valid_spawns_mask: board_data.valid_spawns_mask,
            factory_occupancy,
            factories_per_team: board_data.factories_per_team,
        }
    }
    #[inline]
    pub fn rubble(&self, pos: &Pos) -> u64 {
        self.rubble[pos.0 as usize][pos.1 as usize]
    }
    #[inline]
    pub fn iter_valid_spawns(&self) -> impl Iterator<Item = Pos> + '_ {
        self.valid_spawns_mask
            .iter()
            .enumerate()
            .flat_map(|(i, y_vals)| {
                y_vals
                    .iter()
                    .enumerate()
                    .filter(|(_, val)| **val)
                    .map(move |(j, _)| (i as i64, j as i64))
            })
    }
}
