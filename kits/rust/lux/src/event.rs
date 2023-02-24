use crate::config::Config;
use crate::observation::Observation;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct EventInfo {
    pub env_cfg: Config,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Event {
    pub obs: Observation,
    pub step: u64,
    #[serde(rename = "remainingOverageTime")]
    pub remaining_overage_time: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub info: Option<EventInfo>,
    pub player: String,
}
