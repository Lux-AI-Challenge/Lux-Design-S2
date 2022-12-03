package com.luxai.lux;

public class Factory {

    public static final int BUILD_LIGHT = 0;
    public static final int BUILD_HEAVY = 1;
    public static final int WATER = 2;

    public static final int X = 0;
    public static final int Y = 1;

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

    public int waterCost(Obs obs, Environment environment) {
        int lichenCounter = 0;
        for (int[] row : obs.board.lichen_strains) {
            for (int lichenStrain : row) {
                if (lichenStrain == this.strain_id)
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
