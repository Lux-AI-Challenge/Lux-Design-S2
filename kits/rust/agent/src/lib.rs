use lux::{Agent, SetupAction, State, UnitActions};

pub struct RandomAgent;

impl Agent for RandomAgent {
    fn setup(&mut self, _: &State) -> Option<SetupAction> {
        None
    }
    fn act(&mut self, _: &State) -> UnitActions {
        vec![].into_iter().collect()
    }
}
