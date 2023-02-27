use crate::action::{Direction, UnitActionCommand};
use crate::cargo::Cargo;
use crate::config::RobotTypeConfig;
use crate::state::State;
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

    /// pos cannot be negative but keeping signed to avoid casting for delta
    /// computations
    pub pos: (i64, i64),
    pub cargo: Cargo,
    pub action_queue: Vec<UnitActionCommand>,
}

impl Robot {
    // TODO(seamooo) cache below rather than allocate every time
    #[inline]
    pub fn agent_id(&self) -> String {
        format!("player_{}", self.team_id)
    }

    #[inline]
    pub fn action_queue_cost(&self, state: &State) -> u64 {
        self.cfg(state).action_queue_power_cost
    }

    #[inline]
    pub fn cfg<'a>(&self, state: &'a State) -> &'a RobotTypeConfig {
        match self.unit_type {
            RobotType::Light => &state.env_cfg.robots.light,
            RobotType::Heavy => &state.env_cfg.robots.heavy,
        }
    }

    /// Calculates the cost for a move in the given direction
    ///
    /// If a move is impossible, this will return `None`
    pub fn move_cost(&self, state: &State, direction: &Direction) -> Option<u64> {
        let target_pos = {
            let (x, y) = direction.to_pos();
            (self.pos.0 + x, self.pos.1 + y)
        };
        if target_pos.0 < 0
            || target_pos.0 >= state.board.rubble.len() as i64
            || target_pos.1 < 0
            || target_pos.1 >= state.board.rubble[0].len() as i64
        {
            return None;
        }
        if let Some(strain_id) =
            &state.board.factory_occupancy[target_pos.0 as usize][target_pos.1 as usize]
        {
            if state
                .teams
                .get(&self.agent_id())
                .unwrap()
                .factory_strains
                .iter()
                .any(|x| x == strain_id)
            {
                return None;
            }
        }
        let rubble = state.board.rubble(&target_pos);
        let cfg = self.cfg(state);
        let rv = cfg.move_cost + f64::floor(cfg.rubble_movement_cost * rubble as f64) as u64;
        Some(rv)
    }
    #[inline(always)]
    pub fn dig_cost(&self, state: &State) -> u64 {
        self.cfg(state).dig_cost
    }
    #[inline(always)]
    pub fn self_destruct_cost(&self, state: &State) -> u64 {
        self.cfg(state).self_destruct_cost
    }
}
