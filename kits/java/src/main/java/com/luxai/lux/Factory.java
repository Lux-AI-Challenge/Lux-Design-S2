package com.luxai.lux;

import com.fasterxml.jackson.annotation.JsonProperty;

public class Factory {

    public static final int BUILD_LIGHT = 0;
    public static final int BUILD_HEAVY = 1;
    public static final int WATER = 2;

    public static final int X = 0;
    public static final int Y = 1;

    public int[] pos;
    public int power;
    @JsonProperty("strain_id")
    public int strainId;
    @JsonProperty("team_id")
    public int teamId;

    @JsonProperty("unit_id")
    public String unitId;
    public Cargo cargo;

    public boolean canBuildHeavy(Environment environment) {
        RobotInfo robot = environment.ROBOTS.get("HEAVY");
        return this.power >= robot.POWER_COST && this.cargo.metal >= robot.METAL_COST;
    }

    public boolean canBuildLight(Environment environment) {
        RobotInfo robot = environment.ROBOTS.get("LIGHT");
        return this.power >= robot.POWER_COST && this.cargo.metal >= robot.METAL_COST;
    }

    public int waterCost(Obs obs, Environment environment) {
        int lichenCounter = 0;
        for (int[] row : obs.board.lichen_strains) {
            for (int lichenStrain : row) {
                if (lichenStrain == this.strainId)
                    lichenCounter++;
            }
        }
        return (int) (Math.ceil(lichenCounter / 10.0) * environment.LICHEN_WATERING_COST_FACTOR);
    }

    public boolean canWater(Obs obs, Environment environment) {
        return this.cargo.water >= waterCost(obs, environment);
    }

    public boolean isFactoryArea(int x, int y) {
        return Math.max(Math.abs(this.pos[X] - x), Math.abs(this.pos[Y] - y)) <= 1;
    }
}
