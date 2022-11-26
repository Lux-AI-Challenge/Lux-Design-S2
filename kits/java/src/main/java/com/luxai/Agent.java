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

    //private State state;

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
            Random random = new Random();
            ArrayList<ArrayList<Integer>> mySpawn = this.obs.board.spawns.get(me());
            int randomSpawnIndex = random.nextInt(mySpawn.size());

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

        Map<String, Factory> factories = obs.factories.get(this.me());
        for (String unitId : factories.keySet()) {
            Factory factory = factories.get(unitId);
            if (factory.canBuildHeavy(this.env_cfg))
                unitAction.add(factory.unit_id, Factory.BUILD_HEAVY);
        }

        Map<String, Robot> units = obs.units.get(this.me());
        for (Robot robot : units.values()) {

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

//    public Obs getObs() {
//        return obs;
//    }

//    public int getStep() {
//        return step;
//    }
//
//    public void setObs(Obs obs) {
//        this.obs = obs;
//    }
//
//    public void setStep(int step) {
//        this.step = step;
//    }
//
//    public void setRemainingOverageTime(int remainingOverageTime) {
//        this.remainingOverageTime = remainingOverageTime;
//    }
//
//    public void setPlayer(String player) {
//        this.player = player;
//    }
//
//    public void setEnv_cfg(Environment env_cfg) {
//        this.env_cfg = env_cfg;
//    }
//
//    public void setReward(double reward) {
//        this.reward = reward;
//    }

}
