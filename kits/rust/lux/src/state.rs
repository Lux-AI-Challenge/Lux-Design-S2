//! A module for creating, updating and interacting with state

use crate::board::{Board, BoardError};
use crate::config::Config;
use crate::event::Event;
use crate::factory::Factory;
use crate::robot::Robot;
use crate::team::Team;
use std::collections::HashMap;
use thiserror::Error;

/// Type capturing all errors for the [`state`](crate::state) module
#[derive(Error, Debug, Clone)]
pub enum StateError {
    /// Expected an Event::InitEvent variant
    #[error("Event passed was not an initial event")]
    NotInitialEvent,

    /// Expected an Event::DeltaEvent variant
    #[error("Event passed was not a delta event")]
    NotDeltaEvent,

    /// A map of player_id to `T` could not find the corresponding
    /// player_id
    ///
    /// This is almost always an internal error
    #[error("Player id was not present in the collection")]
    PlayerNotFound,

    /// Error whilst updating state from a board delta
    #[error("Error while updating board ({0})")]
    BoardUpdateError(#[from] BoardError),
}

/// Struct representing both the initial config and current game state
///
/// # Performance notes
///
/// Although this struct is cloneable, it is quite large, especially the
/// `board` member. As such, minimising clones, or using a compressed representation,
/// would be preferable for performance.
///
/// # Note
///
/// This must be created using [`State::from_initial_event`], and should be updated
/// by [`State::update_from_delta_event`] such that expected checks are performed.
/// As such this has been marked as `non_exaustive`, and should be treated as
/// immutable outside of [`State::update_from_delta_event`]
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct State {
    /// Signed step indicating game phase and step
    ///
    /// recommended to not serialize this if representing state as a vector,
    /// but instead serialize the sign as 0 / 1
    pub env_steps: i64,

    /// Unsigned step especially useful for indicating turns
    pub step: u64,

    /// Config for the game set on init
    pub env_cfg: Config,

    /// Current board state
    pub board: Board,

    /// player -> unit_ids -> units mapping
    pub units: HashMap<String, HashMap<String, Robot>>,

    /// player -> factory_ids -> factories mapping
    pub factories: HashMap<String, HashMap<String, Factory>>,

    /// player -> team mapping
    pub teams: HashMap<String, Team>,

    /// player id set on init
    pub player: String,

    /// Amount of time left
    pub remaining_overage_time: i64,
}

impl State {
    /// Creates a state struct from an initial event
    pub fn from_init_event(event: Event) -> Result<Self, StateError> {
        if let Event::InitEvent {
            obs,
            step,
            remaining_overage_time,
            info,
            player,
        } = event
        {
            let env_steps = obs.real_env_steps;
            let env_cfg = info.env_cfg;
            let mut factories = obs.factories;
            let board = Board::from_data_and_factories(obs.board, &factories);
            let mut units = obs.units;
            let teams = obs.teams;
            units.entry(player.clone()).or_default();
            factories.entry(player.clone()).or_default();
            Ok(Self {
                env_steps,
                step,
                env_cfg,
                board,
                units,
                factories,
                teams,
                player,
                remaining_overage_time,
            })
        } else {
            Err(StateError::NotInitialEvent)
        }
    }

    /// Updates the mutable portions of state from the given [`Event`]
    pub fn update_from_delta_event(&mut self, event: Event) -> Result<(), StateError> {
        if let Event::DeltaEvent {
            obs,
            step,
            remaining_overage_time,
            ..
        } = event
        {
            self.step = step;
            self.env_steps = obs.real_env_steps;
            self.board.update_from_delta(obs.board)?;
            self.units = obs.units;
            self.factories = obs.factories;
            self.teams = obs.teams;
            self.units.entry(self.player.clone()).or_default();
            self.factories.entry(self.player.clone()).or_default();
            self.remaining_overage_time = remaining_overage_time;
            Ok(())
        } else {
            Err(StateError::NotDeltaEvent)
        }
    }

    /// Evaluates if the player owning this state can place a factory
    /// this turn.
    ///
    /// This is impossible during the initialization event and will error
    pub fn can_place_factory(&self) -> Result<bool, StateError> {
        let bit = if self.my_team()?.place_first { 1 } else { 0 };
        Ok(self.step & 0x1 == bit)
    }

    /// Retrieves the team for the owner of this state
    ///
    /// If this is used during the initialization event the info
    /// will not be populated and this will error
    #[inline(always)]
    pub fn my_team(&self) -> Result<&Team, StateError> {
        self.teams
            .get(&self.player)
            .ok_or(StateError::PlayerNotFound)
    }

    /// Retrieves the factories for the owner of this state
    #[inline(always)]
    pub fn my_factories(&self) -> &HashMap<String, Factory> {
        self.factories.get(&self.player).unwrap()
    }

    /// Retrieves the units (i.e. robots) for the owner of this state
    #[inline(always)]
    pub fn my_units(&self) -> &HashMap<String, Robot> {
        self.units.get(&self.player).unwrap()
    }
}
