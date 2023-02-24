use crate::board::Board;
use crate::factory::Factory;
use crate::robot::Robot;
use crate::team::Team;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
pub struct Observation {
    pub units: HashMap<String, HashMap<String, Robot>>,
    pub factories: HashMap<String, HashMap<String, Factory>>,
    pub board: Board,
    pub teams: HashMap<String, Team>,
    /// can be negative due to there being two phases of gameplay
    pub real_env_steps: i64,
    pub global_id: u64,
}
