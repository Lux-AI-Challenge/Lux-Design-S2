//! A module for representing the messages for clients to respond to

use crate::config::Config;
use crate::observation::Observation;
use serde::{Deserialize, Serialize};

/// Utility struct for mimicing the JSON format for config
#[derive(Serialize, Deserialize, Debug)]
pub struct EventInfo {
    /// Environment config
    pub env_cfg: Config,
}

/// Message for a client to receive
#[derive(Serialize, Deserialize, Debug)]
pub struct Event {
    /// Observed mutated state
    pub obs: Observation,

    /// Unsigned step such that game phase information is lost
    pub step: u64,

    /// Remaining time bank for overtime usage
    #[serde(rename = "remainingOverageTime")]
    pub remaining_overage_time: i64,

    /// Config present on initial event
    #[serde(skip_serializing_if = "Option::is_none")]
    pub info: Option<EventInfo>,

    /// Identifier for the current player receiving the event
    pub player: String,
}
