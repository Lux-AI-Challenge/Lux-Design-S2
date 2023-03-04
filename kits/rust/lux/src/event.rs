//! A module for representing the messages for clients to respond to

use crate::config::Config;
use crate::observation::{Observation, ObservationDelta};
use serde::{Deserialize, Serialize};

/// Utility struct for mimicing the JSON format for config
#[derive(Serialize, Deserialize, Debug)]
pub struct InitEventInfo {
    /// Environment config
    pub env_cfg: Config,
}

/// Message for a client to receive
#[derive(Serialize, Deserialize, Debug)]
#[serde(untagged)]
pub enum Event {
    /// Event for initialization
    InitEvent {
        /// Observed mutated state
        obs: Observation,

        /// Unsigned step such that game phase information is lost but event order
        /// is communicated
        step: u64,

        /// Remaining time bank for overtime usage
        #[serde(rename = "remainingOverageTime")]
        remaining_overage_time: i64,

        /// Config present on initial event
        info: Box<InitEventInfo>,

        /// Identifier for the current player receiving the event
        player: String,
    },
    /// Event for incremental updates
    DeltaEvent {
        /// Observed mutated state delta
        obs: ObservationDelta,

        /// Unsigned step such that game phase information is lost but event order
        /// is communicated
        step: u64,

        /// Remaining time bank for overtime usage
        #[serde(rename = "remainingOverageTime")]
        remaining_overage_time: i64,

        /// Identifier for the current player receiving the event
        player: String,
    },
}
