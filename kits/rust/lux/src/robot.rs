//! A module for handling and interacting with robot units

use crate::action::{Direction, RobotActionCommand};
use crate::cargo::Cargo;
use crate::config::RobotTypeConfig;
use crate::state::State;
use crate::Pos;
use serde::{Deserialize, Serialize};

/// Denotes robot type ie one of heavy or light
#[allow(missing_docs)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RobotType {
    #[serde(rename = "LIGHT")]
    Light,
    #[serde(rename = "HEAVY")]
    Heavy,
}

/// A representation of a Robot
///
/// A robot is either heavy or light and able to move, dig rubble, dig resources,
/// dig lichen, transfer resources, self destruct and potentially battle other robots
/// by [occupying the same tile](https://www.lux-ai.org/specs-s2#movement-collisions-and-rubble)
///
/// For more information see the [spec](https://www.lux-ai.org/specs-s2#robots)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Robot {
    /// Id discriminating the team Factory belongs to
    ///
    /// valid values are 0 or 1
    pub team_id: u64,

    /// Unique identfier for the factory
    ///
    /// Will be of the form "unit_<unique_index>"
    pub unit_id: String,

    /// Current unit power
    pub power: u64,

    /// Robot type ie one of heavy or light
    pub unit_type: RobotType,

    /// (x, y) position of tile currently occupied by the unit
    pub pos: Pos,

    /// Current materials stored by the unit
    pub cargo: Cargo,

    /// Current state of the action_queue, arranged in descending action priority
    /// i.e. the next action to execute will be at index 0
    pub action_queue: Vec<RobotActionCommand>,
}

impl Robot {
    /// Gets the identifier for the robot's team
    #[inline]
    pub fn agent_id(&self) -> String {
        format!("player_{}", self.team_id)
    }

    /// Gets the power cost to queue an action for the robot
    #[inline]
    pub fn action_queue_cost(&self, state: &State) -> u64 {
        self.cfg(state).action_queue_power_cost
    }

    /// Gets the power cost to queue an action for the robot
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
        if matches!(direction, Direction::Center) {
            return Some(0);
        }
        let target_pos = self.pos + direction.to_pos();
        if target_pos.0 < 0
            || target_pos.0 >= state.board.x_len() as i64
            || target_pos.1 < 0
            || target_pos.1 >= state.board.y_len() as i64
        {
            return None;
        }
        if let Some(strain_id) = &state.board.factory_occupancy(&target_pos) {
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

    /// Gets the power cost to perform a dig action
    #[inline(always)]
    pub fn dig_cost(&self, state: &State) -> u64 {
        self.cfg(state).dig_cost
    }

    /// Gets the power cost to perform a self destruct action
    #[inline(always)]
    pub fn self_destruct_cost(&self, state: &State) -> u64 {
        self.cfg(state).self_destruct_cost
    }

    /// Checks if the robot has enough power to queue a dig move
    #[inline(always)]
    pub fn can_dig(&self, state: &State) -> bool {
        self.dig_cost(state) + self.action_queue_cost(state) <= self.power
    }

    /// Checks if the robot has enough power to queue a transfer move
    #[inline(always)]
    pub fn can_transfer(&self, state: &State) -> bool {
        self.action_queue_cost(state) <= self.power
    }

    /// Checks if the robot can move in the given direction
    #[inline(always)]
    pub fn can_move(&self, state: &State, direction: &Direction) -> bool {
        match self.move_cost(state, direction) {
            Some(cost) => cost + self.action_queue_cost(state) <= self.power,
            None => false,
        }
    }
}
