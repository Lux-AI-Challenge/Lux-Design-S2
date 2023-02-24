use agent::RandomAgent;
use lux::{Agent, Event, State};
use serde_json::de::from_reader;
use std::io;

// TODO(seamooo) what is the return value
fn exec_step<A: Agent>(agent: &mut A, state: &State) {
    agent.setup(state);
    agent.act(state);
}

fn run<A: Agent>(agent: &mut A) {
    let init_event: Event = from_reader(io::stdin()).unwrap();
    let mut state = State::from_init_event(init_event);
    exec_step(agent, &state);
    while let Ok(event) = from_reader(io::stdin()) {
        state.update_from_event(event);
        exec_step(agent, &state);
    }
}

fn main() {
    // swap out agents here
    let mut agent = RandomAgent {};
    run(&mut agent);
}
