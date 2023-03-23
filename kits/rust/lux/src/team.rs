//! A module for serializing, deserializing and handling Team values

use serde::{Deserialize, Serialize};

/// Faction for an agent
///
/// # Clarification
/// Uncertain if other values are acceptable for this
#[allow(missing_docs)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Faction {
    AlphaStrike,
    MotherMars,
    TheBuilders,
    FirstMars,
}

/// A struct to store agent-wide values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Team {
    /// Id uniquely identifying the team
    ///
    /// valid values are 0 or 1
    pub team_id: u64,

    /// Faction assigned to the agent
    pub faction: Faction,

    /// Total water available unassigned to a factory
    pub water: u64,

    /// Total metal available unassigned to a factory
    pub metal: u64,

    /// Count for remaining factories to spawn
    pub factories_to_place: u64,

    /// Array of unique lichen strains produced by the team's factories
    pub factory_strains: Vec<u64>,

    /// Did this team win the bid and can place first
    pub place_first: bool,

    /// Value of the team's bid
    pub bid: u64,
}
