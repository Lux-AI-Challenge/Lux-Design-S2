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
}

impl State {
    pub fn from_init_event(event: Event) -> Self {
        if event.step != 0 {
            panic!("event was not an initial event");
        }
        let step = event.step;
        let env_steps = event.obs.real_env_steps;
        let env_cfg = event.info.unwrap().env_cfg;
        let board = event.obs.board;
        let units = event.obs.units;
        let factories = event.obs.factories;
        let teams = event.obs.teams;
        let player = event.player;
        Self {
            env_steps,
            step,
            env_cfg,
            board,
            units,
            factories,
            teams,
            player,
        }
    }

    pub fn update_from_event(&mut self, event: Event) {
        self.step = event.step;
        self.env_steps = event.obs.real_env_steps;
        self.board = event.obs.board;
        self.units = event.obs.units;
        self.factories = event.obs.factories;
        self.teams = event.obs.teams;
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
    #[inline(always)]
    pub fn my_team(&self) -> &Team {
        self.teams.get(&self.player).unwrap()
    }
    #[inline(always)]
    pub fn my_factories(&self) -> &HashMap<String, Factory> {
        self.factories.get(&self.player).unwrap()
    }
    #[inline(always)]
    pub fn my_units(&self) -> &HashMap<String, Robot> {
        self.units.get(&self.player).unwrap()
    }
}
