//! A module for creating, updating and interacting with state

use crate::board::Board;
use crate::config::Config;
use crate::event::Event;
use crate::factory::Factory;
use crate::robot::Robot;
use crate::team::Team;
use std::collections::HashMap;

/// Struct representing both the initial config and current game state
///
/// # Performance notes
///
/// Although this struct is cloneable, it is quite large, especially the
/// `board` member. As such, minimising clones, or using a compressed representation,
/// would be preferable for performance.
#[derive(Clone, Debug)]
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
    ///
    /// # Panics
    ///
    /// Panics if the event sent was not an [`Event::InitEvent`] variant
    pub fn from_init_event(event: Event) -> Self {
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
            let factories = obs.factories;
            let board = Board::from_data_and_factories(obs.board, &factories);
            let units = obs.units;
            let teams = obs.teams;
            Self {
                env_steps,
                step,
                env_cfg,
                board,
                units,
                factories,
                teams,
                player,
                remaining_overage_time,
            }
        } else {
            panic!("event was not an initial event");
        }
    }

    /// Updates the mutable portions of state from the given [`Event`]
    ///
    /// # Panics
    ///
    /// Panics if the event sent was not an [`Event::DeltaEvent`] variant
    pub fn update_from_delta_event(&mut self, event: Event) {
        if let Event::DeltaEvent {
            obs,
            step,
            remaining_overage_time,
            ..
        } = event
        {
            // TODO(seamooo) assert that this is correct for units / factories
            self.step = step;
            self.env_steps = obs.real_env_steps;
            self.board.update_from_delta(obs.board);
            self.units = obs.units;
            self.factories = obs.factories;
            self.teams = obs.teams;
            self.remaining_overage_time = remaining_overage_time;
        } else {
            panic!("event was not a delta event");
        }
    }

    /// Evaluates if the player owning this state can place a factory
    /// this turn.
    pub fn can_place_factory(&self) -> bool {
        let bit = if self.teams.get(&self.player).unwrap().place_first {
            1
        } else {
            0
        };
        self.step & 0x1 == bit
    }

    /// Retrieves the team for the owner of this state
    #[inline(always)]
    pub fn my_team(&self) -> &Team {
        self.teams.get(&self.player).unwrap()
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
