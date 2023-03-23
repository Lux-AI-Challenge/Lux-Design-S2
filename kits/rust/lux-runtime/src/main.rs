use agent::RandomAgent;
use anyhow::Result;
use log::{debug, error};
use lux::{Agent, Event, State};
use serde_json::json;
use std::io::{self, Write};

fn exec_step<A: Agent>(agent: &mut A, state: &State) -> serde_json::Value {
    if state.env_steps <= 0 {
        agent
            .setup(state)
            .map(|x| serde_json::to_value(x).unwrap())
            .unwrap_or(json!({}))
    } else {
        serde_json::to_value(agent.act(state)).unwrap()
    }
}

fn next_event() -> Result<Event> {
    debug!("receiving next event");
    let line = {
        let mut rv = String::new();
        io::stdin().read_line(&mut rv)?;
        rv
    };
    let rv: Event = serde_json::from_str(line.as_str())?;
    Ok(rv)
}

fn send_msg(msg_val: &serde_json::Value) -> Result<()> {
    debug!("sending message: {msg_val}");
    let msg = format!("{msg_val}\n");
    io::stdout().write_all(msg.as_bytes())?;
    Ok(())
}

fn run<A: Agent>(agent: &mut A) -> Result<()> {
    let init_event: Event = next_event()?;
    let mut state = State::from_init_event(init_event)?;
    send_msg(&exec_step(agent, &state))?;
    while let Ok(event) = next_event() {
        state.update_from_delta_event(event)?;
        send_msg(&exec_step(agent, &state))?;
    }
    Ok(())
}

fn main() {
    env_logger::init();
    // swap out agents here
    let mut agent = RandomAgent::default();
    if let Err(err) = run(&mut agent) {
        error!("{}", err);
    }
}
