//! Handles board queries, and serializing / deserializing board data.

use crate::factory::Factory;
use crate::utils::RectMat;
use crate::Pos;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Type capturing all errors for the [`board`](crate::board) module
#[derive(Error, Debug, Clone)]
pub enum BoardError {
    /// Found a key not of the form "x,y" for one of the delta
    /// fields in [`BoardDelta`]
    #[error("Delta key was not of the form \"x,y\"")]
    BadDeltaKeyFormat,
}

/// Utility struct for communicating [`Board`]
#[derive(Debug, Deserialize, Serialize)]
pub struct BoardData {
    rubble: Vec<Vec<u64>>,
    ice: Vec<Vec<u64>>,
    ore: Vec<Vec<u64>>,
    lichen: Vec<Vec<u64>>,
    lichen_strains: Vec<Vec<i64>>,
    valid_spawns_mask: Vec<Vec<bool>>,
    factories_per_team: u64,
}

impl From<Board> for BoardData {
    fn from(val: Board) -> Self {
        <Self as From<&'_ Board>>::from(&val)
    }
}

type FiveTuplePairs<T0, T1, T2, T3, T4> = (T0, (T1, (T2, (T3, T4))));
type FiveTuplePairsVecVec<T0, T1, T2, T3, T4> =
    FiveTuplePairs<Vec<Vec<T0>>, Vec<Vec<T1>>, Vec<Vec<T2>>, Vec<Vec<T3>>, Vec<Vec<T4>>>;

impl From<&'_ Board> for BoardData {
    fn from(val: &Board) -> Self {
        let (rubble, (ice, (ore, (lichen, lichen_strains)))): FiveTuplePairsVecVec<_, _, _, _, _> =
            (0..val.x_len())
                .map(|row_idx| {
                    val.tiles
                        .iter_row(row_idx)
                        .map(|tile| {
                            let lichen = tile.lichen.as_ref().map(|x| x.count).unwrap_or_default();
                            let lichen_strain =
                                tile.lichen.as_ref().map(|x| x.strain as i64).unwrap_or(-1);
                            (tile.rubble, (tile.ice, (tile.ore, (lichen, lichen_strain))))
                        })
                        .unzip()
                })
                .unzip();
        let valid_spawns_mask: Vec<Vec<_>> = (0..val.x_len())
            .map(|row_idx| val.valid_spawns_mask.iter_row(row_idx).cloned().collect())
            .collect();
        Self {
            rubble,
            ice,
            ore,
            lichen,
            lichen_strains,
            valid_spawns_mask,
            factories_per_team: val.factories_per_team,
        }
    }
}

/// Utility struct for serializing [`Board`]
#[derive(Serialize)]
pub(crate) struct BoardDataRef<'a> {
    rubble: &'a Vec<Vec<u64>>,
    ice: &'a Vec<Vec<u64>>,
    ore: &'a Vec<Vec<u64>>,
    lichen: &'a Vec<Vec<u64>>,
    lichen_strains: &'a Vec<Vec<i64>>,
    valid_spawns_mask: &'a Vec<Vec<bool>>,
    factories_per_team: &'a u64,
}

/// Utility struct for serializing and deserializing ObservationDelta
#[derive(Debug, Deserialize, Serialize)]
pub struct BoardDelta {
    /// Key value pairs denoting deltas between two rubble boards,
    ///
    /// keys are of the form "{x},{y}" where x and y are grid coordinates.
    /// e.g. "1,2"
    pub rubble: HashMap<String, u64>,

    /// Key value pairs denoting deltas between two lichen boards,
    ///
    /// keys are of the form "{x},{y}" where x and y are grid coordinates.
    /// e.g. "1,2"
    pub lichen: HashMap<String, u64>,

    /// Key value pairs denoting deltas between two lichen strain boards,
    ///
    /// keys are of the form "{x},{y}" where x and y are grid coordinates.
    /// e.g. "1,2"
    ///
    /// This is used to determine if lichen is present
    pub lichen_strains: HashMap<String, i64>,

    /// An optional valid spawns mask if the valid spawns have been updated
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_spawns_mask: Option<Vec<Vec<bool>>>,
}

impl BoardDelta {
    fn key_to_idx(key: &str) -> Result<(usize, usize), BoardError> {
        let mut idx_iter = key.split(',');
        move || -> Result<(usize, usize), ()> {
            let rv: (usize, usize) = (
                idx_iter.next().ok_or(())?.parse().map_err(|_| ())?,
                idx_iter.next().ok_or(())?.parse().map_err(|_| ())?,
            );
            Ok(rv)
        }()
        .map_err(|_| BoardError::BadDeltaKeyFormat)
    }
}

/// Struct denoting the lichen strain and count used in [`Tile`]
#[derive(Clone, Debug)]
pub struct Lichen {
    /// Strain identifier unique to a factory
    pub strain: u64,

    /// Lichen count present for the tile
    pub count: u64,
}

/// Information on the state of a given tile of a board
#[derive(Clone, Debug)]
pub struct Tile {
    /// Rubble count present for the tile
    pub rubble: u64,

    /// Ice count present for the tile
    pub ice: u64,

    /// Ore count present for the tile
    pub ore: u64,

    /// Lichen count present for the tile
    pub lichen: Option<Lichen>,

    /// Strain id for a factory if the tile is occupied by a factory
    pub factory_occupancy: Option<u64>,
}

/// Gameplay board representation
///
/// Matrices in this representation are ordered such that they can be indexed by
/// x, y coordinates e.g. `board.rubble.get((x,y))` would give the rubble at
/// coordinate (x, y)
#[derive(Clone, Debug)]
pub struct Board {
    dims: (usize, usize),

    /// m x n matrix of values denoting the current state of each tile of the board
    pub tiles: RectMat<Tile>,

    /// m x n matrix of values denoting if a spawn is valid at a tile
    pub valid_spawns_mask: RectMat<bool>,

    /// Number of factories for each team
    pub factories_per_team: u64,
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
            let mut rv = RectMat::<Option<u64>>::from_dims_default((m, n));
            for factory_infos in factories.values() {
                for factory in factory_infos.values() {
                    let (x_range, y_range) = factory.occupied_range();
                    for i in x_range {
                        for j in y_range.clone() {
                            let idx = (i as usize, j as usize);
                            *rv.get_mut_unchecked(idx) = Some(factory.strain_id);
                        }
                    }
                }
            }
            rv
        };
        let lichen_iter = board_data
            .lichen_strains
            .into_iter()
            .flatten()
            .zip(board_data.lichen.into_iter().flatten())
            .map(|(val, count)| match val.cmp(&0) {
                std::cmp::Ordering::Less => None,
                _ => Some(Lichen {
                    strain: val as u64,
                    count,
                }),
            });
        let tiles_iter = board_data
            .rubble
            .into_iter()
            .flatten()
            .zip(board_data.ice.into_iter().flatten())
            .zip(board_data.ore.into_iter().flatten())
            .zip(lichen_iter)
            .zip(factory_occupancy.into_iter())
            .map(|((((rubble, ice), ore), lichen), factory_occupancy)| Tile {
                rubble,
                ice,
                ore,
                lichen,
                factory_occupancy,
            });
        let tiles = RectMat::from_chunkable_iter(tiles_iter, n).unwrap();
        Self {
            dims: (m, n),
            tiles,
            valid_spawns_mask: board_data.valid_spawns_mask.try_into().unwrap(),
            factories_per_team: board_data.factories_per_team,
        }
    }
    /// Gets the rubble at the given coord
    #[inline]
    pub fn rubble(&self, pos: &Pos) -> u64 {
        self.tiles.get(pos.as_idx()).unwrap().rubble
    }
    /// Gets the ice at the given coord
    #[inline]
    pub fn ice(&self, pos: &Pos) -> u64 {
        self.tiles.get(pos.as_idx()).unwrap().ice
    }
    /// Gets the ore at the given coord
    #[inline]
    pub fn ore(&self, pos: &Pos) -> u64 {
        self.tiles.get(pos.as_idx()).unwrap().ore
    }
    /// Gets the lichen at the given coord
    #[inline]
    pub fn lichen(&self, pos: &Pos) -> u64 {
        self.tiles
            .get(pos.as_idx())
            .unwrap()
            .lichen
            .as_ref()
            .map(|x| x.count)
            .unwrap_or_default()
    }
    /// Gets the lichen strain at the given coord if there is lichen there
    #[inline]
    pub fn lichen_strain(&self, pos: &Pos) -> Option<u64> {
        self.tiles
            .get(pos.as_idx())
            .unwrap()
            .lichen
            .as_ref()
            .map(|x| x.strain)
    }
    /// Gets the strain id of a factory at the given coord if a factory exists
    #[inline]
    pub fn factory_occupancy(&self, pos: &Pos) -> Option<u64> {
        self.tiles.get(pos.as_idx()).unwrap().factory_occupancy
    }
    /// Iterates all positions that are valid spawn locations
    #[inline]
    pub fn iter_valid_spawns(&self) -> impl Iterator<Item = Pos> + '_ {
        self.valid_spawns_mask
            .enumerate()
            .filter(|(_, val)| **val)
            .map(|(idx, _)| Pos(idx.0 as i64, idx.1 as i64))
    }
    /// Gets the span of x coords for the board
    #[inline]
    pub fn x_len(&self) -> usize {
        self.dims.0
    }
    /// Gets the span of y coords for the board
    #[inline]
    pub fn y_len(&self) -> usize {
        self.dims.1
    }

    pub(crate) fn update_from_delta(&mut self, delta: BoardDelta) -> Result<(), BoardError> {
        if let Some(valid_spawns_mask) = delta.valid_spawns_mask {
            self.valid_spawns_mask = valid_spawns_mask.try_into().unwrap();
        }
        for (key, val) in delta.rubble.into_iter() {
            let idx = BoardDelta::key_to_idx(&key)?;
            self.tiles.get_mut(idx).unwrap().rubble = val;
        }
        for (key, val) in delta.lichen_strains.into_iter() {
            let idx = BoardDelta::key_to_idx(&key)?;
            self.tiles.get_mut(idx).unwrap().lichen = match val.cmp(&0) {
                std::cmp::Ordering::Less => None,
                _ => Some(Lichen {
                    strain: val as u64,
                    count: 0,
                }),
            };
        }
        for (key, val) in delta.lichen.into_iter() {
            let idx = BoardDelta::key_to_idx(&key)?;
            if let Some(lichen) = &mut self.tiles.get_mut(idx).unwrap().lichen {
                lichen.count = val;
            }
        }
        Ok(())
    }
}
