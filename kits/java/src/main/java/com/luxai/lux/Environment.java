package com.luxai.lux;

import java.util.Map;

public class Environment {
    public int max_episode_length;
    public int map_size;
    public int verbose;
    public boolean validate_action_space;
    public int max_transfer_amount;
    public int MIN_FACTORIES;
    public int MAX_FACTORIES;
    public int CYCLE_LENGTH;
    public int DAY_LENGTH;
    public int UNIT_ACTION_QUEUE_SIZE;
    public Map<String, RobotInfo> ROBOTS;
    public int MAX_RUBBLE;
    public int FACTORY_RUBBLE_AFTER_DESTRUCTION;
    public int INIT_WATER_METAL_PER_FACTORY;
    public int INIT_POWER_PER_FACTORY;
    public int MIN_LICHEN_TO_SPREAD;
    public int LICHEN_LOST_WITHOUT_WATER;
    public int LICHEN_GAINED_WITH_WATER;
    public int MAX_LICHEN_PER_TILE;
    public int LICHEN_WATERING_COST_FACTOR;
    public boolean BIDDING_SYSTEM;
    public int FACTORY_PROCESSING_RATE_WATER;
    public int ICE_WATER_RATIO;
    public int FACTORY_PROCESSING_RATE_METAL;
    public int ORE_METAL_RATIO;
    public float POWER_LOSS_FACTOR;
    public int POWER_PER_CONNECTED_LICHEN_TILE;
    public int FACTORY_CHARGE;
    public int FACTORY_WATER_CONSUMPTION;

}
