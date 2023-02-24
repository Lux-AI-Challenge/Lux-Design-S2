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

pub use action::{SetupAction, UnitActionCommand, UnitActions};
pub use board::Board;
pub use cargo::Cargo;
pub use config::Config;
pub use event::Event;
pub use factory::Factory;
pub use observation::Observation;
pub use robot::Robot;
pub use state::State;
pub use team::Team;

pub trait Agent {
    fn setup(&mut self, state: &State) -> Option<SetupAction>;
    fn act(&mut self, state: &State) -> UnitActions;
}
