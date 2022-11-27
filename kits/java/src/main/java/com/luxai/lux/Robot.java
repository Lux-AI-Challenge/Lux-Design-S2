package com.luxai.lux;

public class Robot {

    public static final int[][] modeDeltaArray = new int[][]{{0, 0}, {0, -1}, {1, 0}, {0, 1}, {-1, 0}};

    public int[] pos;
    public int team_id;
    public int power;
    public String unit_id;
    public String unit_type;
    public String[] action_queue;
    public Cargo cargo;

    public double getActionQueueCost (Obs obs, Environment environment) {
        int cost = environment.UNIT_ACTION_QUEUE_POWER_COST.get(this.unit_type);
        int currentWeather = obs.weather_schedule[obs.real_env_steps];
        double weatherLossFactor = Weather.powerLossFactor(currentWeather, environment);
        return cost * weatherLossFactor;
    }

    // Сравнить по team_id
    public int getMoveCost(Obs obs, Environment environment, String me, int moveDirection) {
        int[] targetPos = new int[]{this.pos[MoveUtils.X] + modeDeltaArray[moveDirection][MoveUtils.X], this.pos[MoveUtils.Y] + modeDeltaArray[moveDirection][MoveUtils.Y]};
        // Off the map
        if (targetPos[MoveUtils.X] < 0
                || targetPos[MoveUtils.Y] < 0
                || targetPos[MoveUtils.X] >= environment.map_size
                || targetPos[MoveUtils.Y] >= environment.map_size)
            return MoveUtils.MOVE_UNAVAILABLE;
        // On the enemy factory
        for (String player : obs.factories.keySet()) {
            if (!player.equals(me)) {
                for (Factory factory : obs.factories.get(player).values()) {
                    if (factory.isFactoryArea(targetPos[MoveUtils.X], targetPos[MoveUtils.Y]))
                        return MoveUtils.MOVE_UNAVAILABLE;
                }
            }
        }

        int targetRubble = obs.board.rubble[targetPos[MoveUtils.Y]][targetPos[MoveUtils.X]];
        double powerLossFactor = Weather.powerLossFactor(obs.weather_schedule[obs.real_env_steps], environment);
        RobotInfo robotInfo = environment.ROBOTS.get(this.unit_type);
        return (int) Math.ceil((robotInfo.MOVE_COST + robotInfo.RUBBLE_MOVEMENT_COST * targetRubble) * powerLossFactor);
    }

    public int getDigCost(Obs obs, Environment environment) {
        double powerLossFactor = Weather.powerLossFactor(obs.weather_schedule[obs.real_env_steps], environment);
        return (int) Math.ceil(environment.ROBOTS.get(this.unit_type).DIG_COST * powerLossFactor);
    }

    public int getSelfDestructCost(Obs obs, Environment environment) {
        double powerLossFactor = Weather.powerLossFactor(obs.weather_schedule[obs.real_env_steps], environment);
        return (int) Math.ceil(environment.ROBOTS.get(this.unit_type).SELF_DESTRUCT_COST * powerLossFactor);
    }

    public Object move(int dir, boolean repeat) {
        return new Object[]{new Integer[]{0, dir, 0, 0, repeat ? 1 : 0}};
    }

    public Object transfer(int dir, int type, int amount, boolean repeat) {
        return new Object[]{new Integer[]{1, dir, type, amount, repeat ? 1 : 0}};
    }

    public Object pickup(int resourceType, int amount, boolean repeat) {
        return new Object[]{new Integer[]{2, 0, resourceType, amount, repeat ? 1 : 0}};
    }

    public Object dig(boolean repeat) {
        return new Object[]{new Integer[]{3, 0, 0, 0, repeat ? 1 : 0}};
    }

    public Object selfDestruct(boolean repeat) {
        return new Object[]{new Integer[]{4, 0, 0, 0, repeat ? 1 : 0}};
    }

    public Object recharge(int awaitPower, boolean repeat) {
        return new Object[]{new Integer[]{5, 0, 0, awaitPower, repeat ? 1 : 0}};
    }

}
