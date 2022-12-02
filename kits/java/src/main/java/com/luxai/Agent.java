package com.luxai;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.luxai.lux.*;
import com.luxai.lux.action.BidAction;
import com.luxai.lux.action.SpawnAction;
import com.luxai.lux.action.UnitAction;
import com.luxai.lux.objectmapper.Mapper;

import java.util.*;

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
        if (this.obs.teams.get(this.me()).factories_to_place > 0
                && this.isMyTurnToPlaceFactory()) {
            int randomSpawnX = this.random.nextInt(this.obs.board.valid_spawns_mask.length);
            int randomSpawnY = this.random.nextInt(this.obs.board.valid_spawns_mask.length);

            if (!this.obs.board.valid_spawns_mask[randomSpawnX][randomSpawnY]) {
                for (int j = 0; j < this.obs.board.valid_spawns_mask[randomSpawnX].length; j++) {
                    if (this.obs.board.valid_spawns_mask[randomSpawnX][j]) {
                        randomSpawnY = j;
                        break;
                    }
                }
                if (!this.obs.board.valid_spawns_mask[randomSpawnX][randomSpawnY]) {
                    for (int i = 0; i < this.obs.board.valid_spawns_mask.length; i++) {
                        for (int j = 0; j < this.obs.board.valid_spawns_mask[i].length; j++) {
                            if (this.obs.board.valid_spawns_mask[i][j]) {
                                randomSpawnX = i;
                                randomSpawnY = j;
                                break;
                            }
                        }
                    }
                }
            }
            SpawnAction spawnAction = new SpawnAction(new int[]{randomSpawnX, randomSpawnY},
                                                        100,//this.getObs().teams.get(this.me()).metal / 2,
                                                        100//this.getObs().teams.get(this.me()).water / 2
                                                     );
            return Mapper.getJson(spawnAction);
        }
        return null;
    }

    public String act() throws JsonProcessingException {
        UnitAction unitAction = new UnitAction();

        factoryProcessor(unitAction);
        robotProcessor(unitAction);

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
            if (state.obs.real_env_steps < 0)
                this.obs.board.valid_spawns_mask = state.obs.board.valid_spawns_mask;


            if (state.obs.board.rubbleUpdate != null) {
                for (Map.Entry<String, Integer> entry : state.obs.board.rubbleUpdate.entrySet()) {
                    String[] coordinates = entry.getKey().split(",");
                    int x = Integer.parseInt(coordinates[0]);
                    int y = Integer.parseInt(coordinates[1]);
                    this.obs.board.rubble[x][y] = entry.getValue();
                }
            }

            if (state.obs.board.lichenUpdate != null) {
                for (Map.Entry<String, Integer> entry : state.obs.board.lichenUpdate.entrySet()) {
                    String[] coordinates = entry.getKey().split(",");
                    int x = Integer.parseInt(coordinates[0]);
                    int y = Integer.parseInt(coordinates[1]);
                    this.obs.board.lichen[x][y] = entry.getValue();
                }
            }

            if (state.obs.board.lichen_strainsUpdate != null) {
                for (Map.Entry<String, Integer> entry : state.obs.board.lichen_strainsUpdate.entrySet()) {
                    String[] coordinates = entry.getKey().split(",");
                    int x = Integer.parseInt(coordinates[0]);
                    int y = Integer.parseInt(coordinates[1]);
                    this.obs.board.lichen_strains[x][y] = entry.getValue();
                }
            }
        }
    }

    private void factoryProcessor(UnitAction unitAction) {
        Map<String, Factory> myFactories = this.obs.factories.get(this.me());
        for (String unitId : myFactories.keySet()) {
            Factory factory = myFactories.get(unitId);
            if (factory.canBuildHeavy(this.env_cfg))
                unitAction.add(factory.unit_id, Factory.BUILD_HEAVY);
        }
    }

    private void robotProcessor(UnitAction unitAction) {
        Map<String, Robot> units = this.obs.units.get(this.me());
        for (Robot robot : units.values()) {
            int xRobot = robot.pos[MoveUtils.X];
            int yRobot = robot.pos[MoveUtils.Y];
            int factoryDistance = 100000;
            Factory nearestFactory = null;
            // Find nearest factory
            for (Factory factory : this.obs.factories.get(this.me()).values()) {
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
                        if (robot.power > robot.getActionQueueCost(this.obs, this.env_cfg))
                            unitAction.actions.put(robot.unit_id, robot.transfer(factoryDirection, 0, robot.cargo.ice, false));
                    }
                    // Factory long away
                    else {
                        int moveCost = robot.getMoveCost(this.obs, this.env_cfg, this.me(), factoryDirection);
                        if (moveCost != MoveUtils.MOVE_UNAVAILABLE
                                && robot.power >= (moveCost + robot.getActionQueueCost(this.obs, this.env_cfg)))
                            unitAction.actions.put(robot.unit_id, robot.move(factoryDirection, false));
                    }
                }
                // Need to mine recourses
                else {
                    // Find closest ice tile
                    int iceDistance = 100000;
                    int xIce = -1;
                    int yIce = -1;
                    for (int x = 0; x < this.env_cfg.map_size; x++) {
                        for (int y = 0; y < this.env_cfg.map_size; y++) {
                            // Tile has ice
                            if (this.obs.board.ice[x][y] > 0) {
                                boolean isMyFactoryArea = false;
                                Map<String, Factory> myFactories = this.obs.factories.get(this.me());
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
                            if (robot.power >= (robot.getDigCost(this.obs, this.env_cfg) + robot.getActionQueueCost(this.obs, this.env_cfg)))
                                unitAction.actions.put(robot.unit_id, robot.dig(false));
                        }
                        // Ice long away
                        else {
                            int iceDirection = MoveUtils.getDirection(xRobot, yRobot, xIce, yIce);
                            int moveCost = robot.getMoveCost(this.obs, this.env_cfg, this.me(), iceDirection);
                            if (moveCost != MoveUtils.MOVE_UNAVAILABLE
                                    && robot.power >= (moveCost + robot.getActionQueueCost(this.obs, this.env_cfg)))
                                unitAction.actions.put(robot.unit_id, robot.move(iceDirection, false));
                        }
                    }
                }
            }
        }
    }

    public String me() {
        return this.player;
    }

    public boolean isMyTurnToPlaceFactory() {
        if (this.obs.teams.get(this.me()).place_first)
            return (this.step % 2 == 1);
        else
            return (this.step % 2 == 0);
    }

}
