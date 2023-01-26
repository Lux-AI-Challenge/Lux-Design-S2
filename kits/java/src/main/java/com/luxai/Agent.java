package com.luxai;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.luxai.lux.Environment;
import com.luxai.lux.Obs;
import com.luxai.lux.action.BidAction;
import com.luxai.lux.action.SpawnAction;
import com.luxai.lux.action.UnitActions;
import com.luxai.objectmapper.Mapper;
import com.luxai.utils.FactoryProcessor;
import com.luxai.utils.MoveUtils;
import com.luxai.utils.RobotProcessor;

import java.util.Random;

public class Agent {

    private final Random random = new Random(2022);

    public Obs obs;
    public int step;
    public int remainingOverageTime;
    public String player;
    public Environment envConfig;

    public String earlySetup() throws JsonProcessingException {
        if (this.step == 0)
            return Mapper.getJson(new BidAction("AlphaStrike", 0));
        if (this.obs.playerToTeam.get(this.player).factoriesToPlace > 0
                && MoveUtils.isMyTurnToPlaceFactory(this.step, this.obs.playerToTeam.get(this.player).placeFirst)) {
            int randomSpawnX = this.random.nextInt(this.obs.board.validSpawnsMask.length);
            int randomSpawnY = this.random.nextInt(this.obs.board.validSpawnsMask.length);

            if (!this.obs.board.validSpawnsMask[randomSpawnX][randomSpawnY]) {
                for (int j = 0; j < this.obs.board.validSpawnsMask[randomSpawnX].length; j++) {
                    if (this.obs.board.validSpawnsMask[randomSpawnX][j]) {
                        randomSpawnY = j;
                        break;
                    }
                }
                if (!this.obs.board.validSpawnsMask[randomSpawnX][randomSpawnY]) {
                    for (int i = 0; i < this.obs.board.validSpawnsMask.length; i++) {
                        for (int j = 0; j < this.obs.board.validSpawnsMask[i].length; j++) {
                            if (this.obs.board.validSpawnsMask[i][j]) {
                                randomSpawnX = i;
                                randomSpawnY = j;
                                break;
                            }
                        }
                    }
                }
            }
            SpawnAction spawnAction = new SpawnAction(new int[]{randomSpawnX, randomSpawnY}, envConfig.INIT_WATER_METAL_PER_FACTORY, envConfig.INIT_WATER_METAL_PER_FACTORY);
            return Mapper.getJson(spawnAction);
        }
        return null;
    }

    public String act() throws JsonProcessingException {
        UnitActions unitActions = new UnitActions();
        unitActions.addActions(FactoryProcessor.getActions(this.obs, this.envConfig, this.player));
        unitActions.addActions(RobotProcessor.getActions(this.obs, this.envConfig, this.player));
        return unitActions.toSystemResponse();
    }

}
