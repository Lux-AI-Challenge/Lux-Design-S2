package com.luxai;

import com.fasterxml.jackson.core.JsonProcessingException;
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

}
