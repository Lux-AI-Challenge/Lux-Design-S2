package com.luxai;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.luxai.lux.*;
import com.luxai.lux.action.BidAction;
import com.luxai.lux.action.SpawnAction;
import com.luxai.lux.action.UnitAction;
import com.luxai.lux.objectmapper.Mapper;

import java.util.ArrayList;
import java.util.Map;
import java.util.Random;

public class Agent {

    private Random random = new Random(2022);

    public Obs obs;
    public int step;
    public int remainingOverageTime;
    public String player;
    public Environment env_cfg;
    public double reward;

    public String early_setup() throws JsonProcessingException {
        if (this.step == 0)
            return Mapper.getJson(new BidAction("AlphaStrike", 0));
        if (this.obs.teams.get(this.me()).factories_to_place > 0) {
            ArrayList<ArrayList<Integer>> mySpawn = this.obs.board.spawns.get(me());
            int randomSpawnIndex = this.random.nextInt(mySpawn.size());

            SpawnAction spawnAction = new SpawnAction(new int[]{mySpawn.get(randomSpawnIndex).get(0), mySpawn.get(randomSpawnIndex).get(1)},
                                                        100,//this.getObs().teams.get(this.me()).metal / 2,
                                                        100//this.getObs().teams.get(this.me()).water / 2
                                                     );

            return Mapper.getJson(spawnAction);
        }
        return null;
    }

    public String act() throws JsonProcessingException {
        UnitAction unitAction = new UnitAction();

        Map<String, Factory> myFactories = obs.factories.get(this.me());
        for (String unitId : myFactories.keySet()) {
            Factory factory = myFactories.get(unitId);
            if (factory.canBuildHeavy(this.env_cfg))
                unitAction.add(factory.unit_id, Factory.BUILD_HEAVY);
        }

        Map<String, Robot> units = obs.units.get(this.me());
        for (Robot robot : units.values()) {
            int xRobot = robot.pos[MoveUtils.X];
            int yRobot = robot.pos[MoveUtils.Y];
            int factoryDistance = 100000;
            Factory nearestFactory = null;
            // Find nearest factory
            for (Factory factory : obs.factories.get(this.me()).values()) {
                int manhattan = MoveUtils.getManhattanDistance(xRobot, yRobot, factory.pos[MoveUtils.X], factory.pos[MoveUtils.Y]);
                if (manhattan < factoryDistance) {
                    factoryDistance = manhattan;
                    nearestFactory = factory;
                }
            }
            if (nearestFactory != null) {
                int factoryDirection = MoveUtils.getDirection(xRobot, yRobot, nearestFactory.pos[MoveUtils.X], nearestFactory.pos[MoveUtils.Y]);
                // Cargo full
                if (robot.cargo.ice > 40) {
                    // Factory orthogonally adjacent
                    if (factoryDistance <= 3) {
                        if (robot.power > robot.getActionQueueCost(obs, env_cfg))
                            unitAction.actions.put(robot.unit_id, robot.transfer(factoryDirection, 0, robot.cargo.ice, false));
                    }
                    // Factory long away
                    else {
                        int moveCost = robot.getMoveCost(obs, env_cfg, this.me(), factoryDirection);
                        if (moveCost != MoveUtils.MOVE_UNAVAILABLE
                                && robot.power >= (moveCost + robot.getActionQueueCost(obs, env_cfg)))
                            unitAction.actions.put(robot.unit_id, robot.move(factoryDirection, false));
                    }
                }
                // Need to mine recourses
                else {
                    // Find closest ice tile
                    int iceDistance = 100000;
                    int xIce = -1;
                    int yIce = -1;
                    for (int x = 0; x < env_cfg.map_size; x++) {
                        for (int y = 0; y < env_cfg.map_size; y++) {
                            // Tile has ice
                            if (obs.board.ice[y][x] > 0) {
                                boolean isMyFactoryArea = false;
                                for (String unitId : myFactories.keySet()) {
                                    Factory factory = myFactories.get(unitId);
                                    if (factory.isFactoryArea(x, y))
                                        isMyFactoryArea = true;
                                }
                                if (!isMyFactoryArea) {
                                    int manhattan = MoveUtils.getManhattanDistance(xRobot, yRobot, x, y);
                                    if (manhattan < iceDistance) {
                                        iceDistance = manhattan;
                                        xIce = x;
                                        yIce = y;
                                    }
                                }
                            }
                        }
                    }
                    // Robot on ice position
                    if (xIce != -1 && yIce != -1) {
                        if (xIce == xRobot && yIce == yRobot) {
                            if (robot.power >= (robot.getDigCost(obs, env_cfg) + robot.getActionQueueCost(obs, env_cfg)))
                                unitAction.actions.put(robot.unit_id, robot.dig(false));
                        }
                        // Ice long away
                        else {
                            int iceDirection = MoveUtils.getDirection(xRobot, yRobot, xIce, yIce);
                            int moveCost = robot.getMoveCost(obs, env_cfg, this.me(), iceDirection);
                            if (moveCost != MoveUtils.MOVE_UNAVAILABLE
                                    && robot.power >= (moveCost + robot.getActionQueueCost(obs, env_cfg)))
                                unitAction.actions.put(robot.unit_id, robot.move(iceDirection, false));
                        }
                    }
                }
            }
        }

        if (unitAction.actions.size() > 0)
            return Mapper.getJson(unitAction.actions);

        return null;
    }

    public void updateState(String json) throws JsonProcessingException {
        ObjectMapper objectMapper = new ObjectMapper();
        // check first step
        if (this.obs == null) {
            json = json
                    .replace("rubble", "rubble_init")
                    .replace("lichen_strains", "strains_init")
                    .replace("lichen", "lichen_init");

            State state = objectMapper.readValue(json, State.class);
            this.obs = state.obs;
            this.step = state.step;
            this.remainingOverageTime = state.remainingOverageTime;
            this.player = state.player;
            this.env_cfg = state.info.env_cfg;
        }
        else {
            State state = objectMapper.readValue(json, State.class);
            this.step = state.step;
            this.remainingOverageTime = state.remainingOverageTime;
            this.obs.teams = state.obs.teams;
            this.obs.real_env_steps = state.obs.real_env_steps;
            this.obs.units.clear(); this.obs.units = state.obs.units;
            this.obs.factories.clear(); this.obs.factories = state.obs.factories;


            if (obs.board.rubbleUpdate != null) {
                for (Map.Entry<String, Integer> entry : obs.board.rubbleUpdate.entrySet()) {
                    String[] coordinates = entry.getKey().split(",");
                    int x = Integer.parseInt(coordinates[0]), y = Integer.parseInt(coordinates[1]);
                    this.obs.board.rubble[y][x] = entry.getValue();
                }
            }

            if (obs.board.lichenUpdate != null) {
                for (Map.Entry<String, Integer> entry : obs.board.lichenUpdate.entrySet()) {
                    String[] coordinates = entry.getKey().split(",");
                    int x = Integer.parseInt(coordinates[0]), y = Integer.parseInt(coordinates[1]);
                    this.obs.board.lichen[y][x] = entry.getValue();
                }
            }

            if (obs.board.lichen_strainsUpdate != null) {
                for (Map.Entry<String, Integer> entry : obs.board.lichen_strainsUpdate.entrySet()) {
                    String[] coordinates = entry.getKey().split(",");
                    int x = Integer.parseInt(coordinates[0]), y = Integer.parseInt(coordinates[1]);
                    this.obs.board.lichen_strains[y][x] = entry.getValue();
                }
            }
        }
    }

    public String me() {
        return this.player;
    }

}
