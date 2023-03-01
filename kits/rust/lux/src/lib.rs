#![warn(missing_docs)]

//! This is a rust implementation inteded for enabling communication and validation
//! of agent actions, as well as providing some helpful utilities for interacting with
//! game rules.
//!
//! **Disclaimer**: Although this kit is heavily documented and well tested. It has
//! been implemented by a third party based on the specification and python
//! implementation. As such, any deviation from the python implementation with
//! regards to documentation or behaviour is unintended and should be logged as a bug

pub mod action;
pub mod board;
pub mod cargo;
pub mod config;
pub mod event;
pub mod factory;
pub mod observation;
pub mod robot;
pub mod state;
pub mod team;
mod utils;

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

// FIXME(seamooo) design choice to separate state into initial conditions and mutable
// vs block struct

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
pub type Pos = (i64, i64);
