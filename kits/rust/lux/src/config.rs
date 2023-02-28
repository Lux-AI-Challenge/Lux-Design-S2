use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub struct RobotTypeConfig {
    pub metal_cost: u64,
    pub power_cost: u64,
    pub cargo_space: u64,
    pub battery_capacity: u64,
    pub charge: u64,
    pub init_power: u64,
    pub move_cost: u64,
    pub rubble_movement_cost: f64,
    pub dig_cost: u64,
    pub dig_rubble_removed: u64,
    pub dig_resource_gain: u64,
    pub dig_lichen_removed: u64,
    pub self_destruct_cost: u64,
    pub rubble_after_destruction: u64,
    pub action_queue_power_cost: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobotConfig {
    #[serde(rename = "HEAVY")]
    pub heavy: RobotTypeConfig,
    #[serde(rename = "LIGHT")]
    pub light: RobotTypeConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub max_episode_length: u64,
    pub map_size: u64,
    pub verbose: u64,
    pub validate_action_space: bool,
    pub max_transfer_amount: u64,
    #[serde(rename = "MIN_FACTORIES")]
    pub min_factories: u64,
    #[serde(rename = "MAX_FACTORIES")]
    pub max_factories: u64,
    #[serde(rename = "CYCLE_LENGTH")]
    pub cycle_length: u64,
    #[serde(rename = "DAY_LENGTH")]
    pub day_length: u64,
    #[serde(rename = "UNIT_ACTION_QUEUE_SIZE")]
    pub unit_action_queue_size: u64,
    #[serde(rename = "MAX_RUBBLE")]
    pub max_rubble: u64,
    #[serde(rename = "FACTORY_RUBBLE_AFTER_DESTRUCTION")]
    pub factory_rubble_after_destruction: u64,
    #[serde(rename = "INIT_WATER_METAL_PER_FACTORY")]
    pub init_water_metal_per_factory: u64,
    #[serde(rename = "INIT_POWER_PER_FACTORY")]
    pub init_power_per_factory: u64,
    #[serde(rename = "MIN_LICHEN_TO_SPREAD")]
    pub min_lichen_to_spread: u64,
    #[serde(rename = "LICHEN_LOST_WITHOUT_WATER")]
    pub lichen_lost_without_water: u64,
    #[serde(rename = "LICHEN_GAINED_WITH_WATER")]
    pub lichen_gained_with_water: u64,
    #[serde(rename = "MAX_LICHEN_PER_TILE")]
    pub max_lichen_per_tile: u64,
    #[serde(rename = "POWER_PER_CONNECTED_LICHEN_TILE")]
    pub power_per_connected_lichen_tile: u64,
    #[serde(rename = "LICHEN_WATERING_COST_FACTOR")]
    pub lichen_watering_cost_factor: f64,
    #[serde(rename = "BIDDING_SYSTEM")]
    pub bidding_system: bool,
    #[serde(rename = "FACTORY_PROCESSING_RATE_WATER")]
    pub factory_processing_rate_water: u64,
    #[serde(rename = "ICE_WATER_RATIO")]
    pub ice_water_ratio: u64,
    #[serde(rename = "FACTORY_PROCESSING_RATE_METAL")]
    pub factory_processing_rate_metal: u64,
    #[serde(rename = "ORE_METAL_RATIO")]
    pub ore_metal_ratio: u64,
    #[serde(rename = "FACTORY_CHARGE")]
    pub factory_charge: u64,
    #[serde(rename = "FACTORY_WATER_CONSUMPTION")]
    pub factory_water_consumption: u64,
    #[serde(rename = "POWER_LOSS_FACTOR")]
    pub power_loss_factor: f64,
    #[serde(rename = "ROBOTS")]
    pub robots: RobotConfig,
}
