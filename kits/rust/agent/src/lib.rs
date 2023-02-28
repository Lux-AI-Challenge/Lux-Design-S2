use lux::{Agent, Faction, SetupAction, State, UnitActions};
use rand::{distributions::Uniform, rngs::SmallRng, Rng, SeedableRng};

pub struct RandomAgent {
    rng: SmallRng,
}

impl Default for RandomAgent {
    fn default() -> Self {
        let rng = SmallRng::from_entropy();
        Self { rng }
    }
}

impl RandomAgent {
    pub fn from_seed(seed: u64) -> Self {
        let rng = <SmallRng as SeedableRng>::seed_from_u64(seed);
        Self { rng }
    }
}

/// An implementation mirroring the logic of the python kit's agent
impl Agent for RandomAgent {
    fn setup(&mut self, state: &State) -> Option<SetupAction> {
        if state.step == 0 {
            Some(SetupAction::Bid {
                faction: Faction::AlphaStrike,
                bid: 0,
            })
        } else if state.can_place_factory() && state.my_team().factories_to_place > 0 {
            let valid_spawns = state.board.iter_valid_spawns().collect::<Vec<_>>();
            let idx: usize = self.rng.sample(Uniform::new(0, valid_spawns.len()));
            Some(SetupAction::Spawn {
                spawn: valid_spawns[idx],
                metal: 150,
                water: 150,
            })
        } else {
            None
        }
    }
    fn act(&mut self, _: &State) -> UnitActions {
        vec![].into_iter().collect()
    }
}
