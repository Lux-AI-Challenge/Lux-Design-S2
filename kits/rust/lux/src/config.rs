//! Environment configuration

use serde::{Deserialize, Serialize};

/// Config shared for both `"HEAVY"` and `"LIGHT"` robot types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub struct RobotTypeConfig {
    /// Metal cost to build a robot of this type
    pub metal_cost: u64,
    /// Power cost to build a robot of this type
    pub power_cost: u64,
    /// Total resource count (other than power) possible to store in this robot
    pub cargo_space: u64,
    /// Total power possible to store in this robot
    pub battery_capacity: u64,
    /// Amount power restored on a per-day basis. Power is not restored during night
    pub charge: u64,
    /// Power that a robot of this type starts with
    pub init_power: u64,
    /// Base power cost to move without rubble impedence
    pub move_cost: u64,
    /// Additional power required per rubble impedence
    pub rubble_movement_cost: f64,
    /// Power required to perform a dig action
    pub dig_cost: u64,
    /// Amount of rubble removed by a dig action
    pub dig_rubble_removed: u64,
    /// Amount of resource gained by a dig action
    pub dig_resource_gain: u64,
    /// Amount of lichen removed by a dig action
    pub dig_lichen_removed: u64,
    /// Amount of poewr required to perform a self destruct action
    pub self_destruct_cost: u64,
    /// Amount of rubble created by a self destruct action
    pub rubble_after_destruction: u64,
    /// Amount of power required to queue an action
    pub action_queue_power_cost: u64,
}

/// Config for all robot types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobotConfig {
    /// Config for `"HEAVY"` robot types
    #[serde(rename = "HEAVY")]
    pub heavy: RobotTypeConfig,
    /// Config for `"LIGHT"` robot types
    #[serde(rename = "LIGHT")]
    pub light: RobotTypeConfig,
}

/// Environment configuration for a single Episode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Number of steps for the length of the game
    pub max_episode_length: u64,

    /// Size used to specify dimensions of a square map
    pub map_size: u64,

    /// Option to enable verbose logging
    pub verbose: u64,

    /// Flag for whether the environment is validating actions
    ///
    /// This is always on for competition servers
    pub validate_action_space: bool,

    /// Maximum amount that can be transfered
    pub max_transfer_amount: u64,

    /// Minimum factories possible in state generation
    #[serde(rename = "MIN_FACTORIES")]
    pub min_factories: u64,

    /// Maximum factories possible in state generation
    #[serde(rename = "MAX_FACTORIES")]
    pub max_factories: u64,

    /// Number of turns in a day / night cycle
    #[serde(rename = "CYCLE_LENGTH")]
    pub cycle_length: u64,

    /// Number of turns constituting the day part of the day / night cycle
    ///
    /// using this in combination with [`Self::cycle_length`] implies night length
    #[serde(rename = "DAY_LENGTH")]
    pub day_length: u64,

    /// Number of items that can exist in a robot's action queue
    #[serde(rename = "UNIT_ACTION_QUEUE_SIZE")]
    pub unit_action_queue_size: u64,

    /// Maximum amount of rubble possible to be present per tile
    #[serde(rename = "MAX_RUBBLE")]
    pub max_rubble: u64,

    /// Amount of rubble created after factory explosion
    #[serde(rename = "FACTORY_RUBBLE_AFTER_DESTRUCTION")]
    pub factory_rubble_after_destruction: u64,

    /// Amount of water and metal given to the player at the beginning of the game
    /// per factory
    #[serde(rename = "INIT_WATER_METAL_PER_FACTORY")]
    pub init_water_metal_per_factory: u64,

    /// Amount of power that a factory begins with
    #[serde(rename = "INIT_POWER_PER_FACTORY")]
    pub init_power_per_factory: u64,

    /// Lichen threshold for a tile to enable lichen spreading
    #[serde(rename = "MIN_LICHEN_TO_SPREAD")]
    pub min_lichen_to_spread: u64,

    /// Amount of lichen lost per unwatered tile per turn
    #[serde(rename = "LICHEN_LOST_WITHOUT_WATER")]
    pub lichen_lost_without_water: u64,

    /// Amount of lichen gained per watered tile per turn
    #[serde(rename = "LICHEN_GAINED_WITH_WATER")]
    pub lichen_gained_with_water: u64,

    /// Maximum value for lichen on a tile
    #[serde(rename = "MAX_LICHEN_PER_TILE")]
    pub max_lichen_per_tile: u64,

    /// Amount of power gained by a factory per connected lichen tile per turn
    #[serde(rename = "POWER_PER_CONNECTED_LICHEN_TILE")]
    pub power_per_connected_lichen_tile: u64,

    /// Cost of watering with a factory calculated by:
    ///
    /// ceil(grow_lichen_tiles_count / [`Self::lichen_watering_cost_factor`])
    #[serde(rename = "LICHEN_WATERING_COST_FACTOR")]
    pub lichen_watering_cost_factor: f64,

    /// Flag to enable bidding system
    #[serde(rename = "BIDDING_SYSTEM")]
    pub bidding_system: bool,

    /// Amount of ice that can be processed by a factory per turn
    #[serde(rename = "FACTORY_PROCESSING_RATE_WATER")]
    pub factory_processing_rate_water: u64,

    /// Units of ice required to produce 1 unit of water
    ///
    /// Water produced calculated by: floor(ice_amount / [`Self::ice_water_ratio`])
    #[serde(rename = "ICE_WATER_RATIO")]
    pub ice_water_ratio: u64,

    /// Amount of ore that can be processed by a factory per turn
    #[serde(rename = "FACTORY_PROCESSING_RATE_METAL")]
    pub factory_processing_rate_metal: u64,

    /// Units of ore required to produce 1 unit of metal
    ///
    /// Metal produced calculated by: floor(ore_amount / [`Self::ore_metal_ratio`])
    #[serde(rename = "ORE_METAL_RATIO")]
    pub ore_metal_ratio: u64,

    /// Amount of poewr gained by a factory per turn regardless of day / night
    #[serde(rename = "FACTORY_CHARGE")]
    pub factory_charge: u64,

    /// Amount of water consumed by a factory per turn
    #[serde(rename = "FACTORY_WATER_CONSUMPTION")]
    pub factory_water_consumption: u64,

    /// Factor used for power loss calculation on unit collision
    ///
    /// New power calculation:
    ///
    /// new_power = old_power - ceil([`Self::power_loss_factor`] * highest_destroyed_power)
    #[serde(rename = "POWER_LOSS_FACTOR")]
    pub power_loss_factor: f64,

    /// Config for all robot types
    #[serde(rename = "ROBOTS")]
    pub robots: RobotConfig,
}
