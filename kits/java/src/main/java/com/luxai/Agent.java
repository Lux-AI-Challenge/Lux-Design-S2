package com.luxai;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.luxai.lux.*;
import com.luxai.lux.action.BidAction;
import com.luxai.lux.action.SpawnAction;
import com.luxai.lux.action.UnitAction;
import com.luxai.lux.objectmapper.Mapper;
import com.luxai.lux.utils.FactoryProcessor;
import com.luxai.lux.utils.MoveUtils;
import com.luxai.lux.utils.RobotProcessor;

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
        if (this.obs.teams.get(this.player).factories_to_place > 0
                && MoveUtils.isMyTurnToPlaceFactory(this.step, this.obs.teams.get(this.player).place_first)) {
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
        UnitAction unitActions = new UnitAction();

        unitActions.addActions(FactoryProcessor.getActions(this.obs, this.env_cfg, this.player));
        unitActions.addActions(RobotProcessor.getActions(this.obs, this.env_cfg, this.player));

        if (unitActions.actions.size() > 0)
            return Mapper.getJson(unitActions.actions);

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

}
