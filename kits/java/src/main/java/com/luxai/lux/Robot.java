package com.luxai.lux;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.luxai.utils.MoveUtils;

public class Robot {

    public static final int[][] deltaArray = new int[][]{{0, 0}, {0, -1}, {1, 0}, {0, 1}, {-1, 0}};

    public int[] pos;
    @JsonProperty("team_id")
    public int teamId;
    public int power;
    @JsonProperty("unit_id")
    public String unitId;
    @JsonProperty("unit_type")
    public String unitType;
    @JsonProperty("action_queue")
    public int[][] actionQueue;
    public Cargo cargo;

    public double getActionQueueCost (Obs obs, Environment environment) {
        return environment.ROBOTS.get(this.unitType).ACTION_QUEUE_POWER_COST;
    }

    public int getMoveCost(Obs obs, Environment environment, String me, int moveDirection) {
        int[] targetPos = new int[]{this.pos[MoveUtils.X] + deltaArray[moveDirection][MoveUtils.X], this.pos[MoveUtils.Y] + deltaArray[moveDirection][MoveUtils.Y]};
        // Off the map
        if (targetPos[MoveUtils.X] < 0
                || targetPos[MoveUtils.Y] < 0
                || targetPos[MoveUtils.X] >= environment.map_size
                || targetPos[MoveUtils.Y] >= environment.map_size)
            return MoveUtils.MOVE_UNAVAILABLE;
        // On the enemy factory
        for (String player : obs.playerToFactories.keySet()) {
            if (!player.equals(me)) {
                for (Factory factory : obs.playerToFactories.get(player).values()) {
                    if (factory.isFactoryArea(targetPos[MoveUtils.X], targetPos[MoveUtils.Y]))
                        return MoveUtils.MOVE_UNAVAILABLE;
                }
            }
        }

        int targetRubble = obs.board.rubble[targetPos[MoveUtils.X]][targetPos[MoveUtils.Y]];
        RobotInfo robotInfo = environment.ROBOTS.get(this.unitType);
        return (int) Math.floor((robotInfo.MOVE_COST + robotInfo.RUBBLE_MOVEMENT_COST * targetRubble));
    }

    public int getDigCost(Obs obs, Environment environment) {
        return environment.ROBOTS.get(this.unitType).DIG_COST;
    }

    public int getSelfDestructCost(Obs obs, Environment environment) {
        return environment.ROBOTS.get(this.unitType).SELF_DESTRUCT_COST;
    }

    public Object move(int dir, int repeat, int iterCount) {
        return new Object[]{new int[]{0, dir, 0, 0, repeat, iterCount}};
    }

    public Object transfer(int dir, int type, int amount, int repeat, int iterCount) {
        return new Object[]{new int[]{1, dir, type, amount, repeat, iterCount}};
    }

    public Object pickup(int resourceType, int amount, int repeat, int iterCount) {
        return new Object[]{new int[]{2, 0, resourceType, amount, repeat, iterCount}};
    }

    public Object dig(int repeat, int iterCount) {
        return new Object[]{new int[]{3, 0, 0, 0, repeat, iterCount}};
    }

    public Object selfDestruct(int repeat, int iterCount) {
        return new Object[]{new int[]{4, 0, 0, 0, repeat, iterCount}};
    }

    public Object recharge(int awaitPower, int repeat, int iterCount) {
        return new Object[]{new int[]{5, 0, 0, awaitPower, repeat, iterCount}};
    }

}
