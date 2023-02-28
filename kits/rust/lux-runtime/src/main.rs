use agent::RandomAgent;
use lux::{Agent, Event, State};
use serde_json::json;
use std::io::{self, Write};

fn exec_step<A: Agent>(agent: &mut A, state: &State) -> serde_json::Value {
    if state.env_steps <= 0 {
        agent
            .setup(state)
            .map(|x| serde_json::to_value(x).unwrap())
            .unwrap_or(json!("{}"))
    } else {
        serde_json::to_value(agent.act(state)).unwrap()
    }
}

fn run<A: Agent>(agent: &mut A) -> io::Result<()> {
    let init_event: Event = serde_json::from_reader(io::stdin()).unwrap();
    let mut state = State::from_init_event(init_event);
    exec_step(agent, &state);
    while let Ok(event) = serde_json::from_reader(io::stdin()) {
        state.update_from_event(event);
        io::stdout().write_all(exec_step(agent, &state).to_string().as_bytes())?;
    }
    Ok(())
}

fn main() {
    // swap out agents here
    let mut agent = RandomAgent::default();
    run(&mut agent).unwrap();
}
