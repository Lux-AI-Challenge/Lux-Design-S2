//! Serializes and deserializes observation data

use crate::board::{BoardData, BoardDelta};
use crate::factory::Factory;
use crate::robot::Robot;
use crate::team::Team;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Observation for change in conditions
/// Current snapshot of the game including all mutated state
#[derive(Debug, Deserialize, Serialize)]
pub struct Observation {
    /// Map of player_id to Map of unit_id to [`Robot`]
    pub units: HashMap<String, HashMap<String, Robot>>,

    /// Map of player_id to Map of unit_id to [`Factory`]
    pub factories: HashMap<String, HashMap<String, Factory>>,

    /// Initial board state
    pub board: BoardData,

    /// Map of player_id to [`Team`]
    pub teams: HashMap<String, Team>,

    /// Can be negative due to there being two phases of gameplay
    pub real_env_steps: i64,

    /// Id to uniquely identify game state
    ///
    /// It is guaranteed that this id would map to this observation and no other
    pub global_id: u64,
}

/// Data struct for incremental observations
#[derive(Serialize, Deserialize, Debug)]
pub struct ObservationDelta {
    /// Map of player_id to Map of unit_id to [`Robot`]
    pub units: HashMap<String, HashMap<String, Robot>>,

    /// Map of player_id to Map of unit_id to [`Factory`]
    pub factories: HashMap<String, HashMap<String, Factory>>,

    /// Board delta for incremental observations
    pub board: BoardDelta,

    /// Map of player_id to [`Team`]
    pub teams: HashMap<String, Team>,

    /// Can be negative due to there being two phases of gameplay
    pub real_env_steps: i64,

    /// Id to uniquely identify game state
    ///
    /// It is guaranteed that this id would map to this observation and no other
    pub global_id: u64,
}
