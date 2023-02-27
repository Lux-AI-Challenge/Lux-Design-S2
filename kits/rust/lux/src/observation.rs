use crate::board::{Board, BoardData, BoardDataRef};
use crate::factory::Factory;
use crate::robot::Robot;
use crate::team::Team;
use serde::{de::Deserializer, ser::Serializer, Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Deserialize)]
struct ObservationData {
    units: HashMap<String, HashMap<String, Robot>>,
    factories: HashMap<String, HashMap<String, Factory>>,
    board: BoardData,
    teams: HashMap<String, Team>,
    real_env_steps: i64,
    global_id: u64,
}

#[derive(Serialize)]
struct ObservationDataRef<'a> {
    units: &'a HashMap<String, HashMap<String, Robot>>,
    factories: &'a HashMap<String, HashMap<String, Factory>>,
    board: BoardDataRef<'a>,
    teams: &'a HashMap<String, Team>,
    real_env_steps: &'a i64,
    global_id: &'a u64,
}

#[derive(Debug)]
pub struct Observation {
    pub units: HashMap<String, HashMap<String, Robot>>,
    pub factories: HashMap<String, HashMap<String, Factory>>,
    pub board: Board,
    pub teams: HashMap<String, Team>,
    /// Can be negative due to there being two phases of gameplay
    pub real_env_steps: i64,
    pub global_id: u64,
}

impl Serialize for Observation {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let ser_val = ObservationDataRef {
            units: &self.units,
            factories: &self.factories,
            board: BoardDataRef::from(&self.board),
            teams: &self.teams,
            real_env_steps: &self.real_env_steps,
            global_id: &self.global_id,
        };
        ser_val.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Observation {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let obs_data = ObservationData::deserialize(deserializer)?;
        let board = Board::from_data_and_factories(obs_data.board, &obs_data.factories);
        let rv = Self {
            units: obs_data.units,
            factories: obs_data.factories,
            board,
            teams: obs_data.teams,
            real_env_steps: obs_data.real_env_steps,
            global_id: obs_data.global_id,
        };
        Ok(rv)
    }
}
