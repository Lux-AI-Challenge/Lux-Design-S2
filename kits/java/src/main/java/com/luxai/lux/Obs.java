package com.luxai.lux;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Map;

public class Obs {
    @JsonProperty("units")
    public Map<String, Map<String, Robot>> playerToUnit;
    @JsonProperty("teams")
    public Map<String,Team> playerToTeam;
    @JsonProperty("factories")
    public Map<String,Map<String, Factory>> playerToFactories;
    @JsonProperty("board")
    public Board board;
    @JsonProperty("real_env_steps")
    public int realEnvSteps;
}
