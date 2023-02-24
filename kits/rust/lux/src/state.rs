use crate::board::Board;
use crate::config::Config;
use crate::event::Event;
use crate::factory::Factory;
use crate::robot::Robot;
use crate::team::Team;
use std::collections::HashMap;

pub struct State {
    pub env_steps: i64,
    pub env_cfg: Config,
    pub board: Board,
    pub units: HashMap<String, HashMap<String, Robot>>,
    pub factories: HashMap<String, HashMap<String, Factory>>,
    pub teams: HashMap<String, Team>,
    pub player: String,
}

impl State {
    pub fn from_init_event(event: Event) -> Self {
        if event.step != 0 {
            panic!("event was not an initial event");
        }
        let env_steps = event.obs.real_env_steps;
        let env_cfg = event.info.unwrap().env_cfg;
        let board = event.obs.board;
        let units = event.obs.units;
        let factories = event.obs.factories;
        let teams = event.obs.teams;
        let player = event.player;
        Self {
            env_steps,
            env_cfg,
            board,
            units,
            factories,
            teams,
            player,
        }
    }
    pub fn update_from_event(&mut self, event: Event) {
        self.board = event.obs.board;
        self.units = event.obs.units;
        self.factories = event.obs.factories;
        self.teams = event.obs.teams;
        unimplemented!();
    }
}
