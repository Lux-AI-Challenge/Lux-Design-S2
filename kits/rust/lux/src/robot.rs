use crate::action::UnitActionCommand;
use crate::cargo::Cargo;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub enum RobotType {
    #[serde(rename = "LIGHT")]
    Light,
    #[serde(rename = "HEAVY")]
    Heavy,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Robot {
    pub team_id: u64,
    pub unit_id: String,
    pub power: u64,
    pub unit_type: RobotType,
    pub pos: [u64; 2],
    pub cargo: Cargo,
    pub action_queue: Vec<UnitActionCommand>,
}
