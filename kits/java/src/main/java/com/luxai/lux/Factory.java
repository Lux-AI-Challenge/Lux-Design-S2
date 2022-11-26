package com.luxai.lux;

public class Factory {

    public static final int BUILD_LIGHT = 0;
    public static final int BUILD_HEAVY = 1;
    public static final int WATER = 2;

    public int[] pos;
    public int power;
    public int strain_id;
    public int team_id;
    public String unit_id;
    public Cargo cargo;

    public boolean canBuildHeavy(Environment environment) {
        RobotInfo robot = environment.ROBOTS.get("HEAVY");
        return this.power >= robot.POWER_COST && this.cargo.metal >= robot.METAL_COST;
    }

    public boolean canBuildLight(Environment environment) {
        RobotInfo robot = environment.ROBOTS.get("LIGHT");
        return this.power >= robot.POWER_COST && this.cargo.metal >= robot.METAL_COST;
    }
}
