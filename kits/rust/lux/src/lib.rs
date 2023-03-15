#![warn(missing_docs)]

//! This is a rust implementation inteded for enabling communication and validation
//! of agent actions, as well as providing some helpful utilities for interacting with
//! game rules.
//!
//! **Disclaimer**: Although this kit is heavily documented and reasonably tested,
//! it has been implemented by a third party based on the specification and python
//! implementation. As such, any deviation from the python implementation with
//! regards to documentation or behaviour is unintended and should be logged as a bug

pub mod action;
pub mod board;
pub mod cargo;
pub mod config;
pub mod error;
pub mod event;
pub mod factory;
pub mod observation;
pub mod robot;
pub mod state;
pub mod team;
pub mod utils;

pub use action::{FactoryAction, RobotActionCommand, SetupAction, UnitAction, UnitActions};
pub use board::Board;
pub use cargo::Cargo;
pub use config::Config;
pub use event::Event;
pub use factory::Factory;
pub use observation::Observation;
pub use robot::Robot;
pub use state::State;
pub use team::{Faction, Team};

use serde::{Deserialize, Serialize};

/// Trait for using an agent in a competition / simulation
pub trait Agent {
    /// Expected to be called during bid phase and setup phase
    fn setup(&mut self, state: &State) -> Option<SetupAction>;

    /// Expected to be called during the action phase
    fn act(&mut self, state: &State) -> UnitActions;
}

/// Position representation used universally
///
/// A signed type has been used such that delta representation is possible
/// with with the same type, however coordinates will all be unsigned
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Pos(pub i64, pub i64);

impl Pos {
    /// # Panics
    ///
    /// Function panics if either parameter is less than 0
    pub fn as_idx(&self) -> (usize, usize) {
        (self.0 as usize, self.1 as usize)
    }
}

impl std::ops::Add<Pos> for Pos {
    type Output = Self;
    fn add(self, rhs: Pos) -> Self::Output {
        Self(self.0 + rhs.0, self.1 + rhs.1)
    }
}

impl std::ops::Sub<Pos> for Pos {
    type Output = Self;
    fn sub(self, rhs: Pos) -> Self::Output {
        Self(self.0 - rhs.0, self.1 - rhs.1)
    }
}

/// General result type that all library function results can transform to
pub type LuxResult<T> = Result<T, error::LuxError>;

/// Module for including essential types in scope
pub mod prelude {
    pub use super::{error::LuxError, Agent, LuxResult, Pos};
}
