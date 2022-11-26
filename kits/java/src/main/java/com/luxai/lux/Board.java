package com.luxai.lux;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.ArrayList;
import java.util.Map;

public class Board {
    // step = 0
    @JsonProperty("rubble_init")
    public int[][] rubble;
    public int[][] ore;
    public int[][] ice;
    @JsonProperty("lichen_init")
    public int[][] lichen;
    @JsonProperty("strains_init")
    public int[][] lichen_strains;
    public Map<String, ArrayList<ArrayList<Integer>>> spawns;

    // step > 0
    @JsonProperty("rubble")
    public Map<String, Integer> rubbleUpdate;
    @JsonProperty("lichen")
    public Map<String, Integer> lichenUpdate;
    @JsonProperty("lichen_strains")
    public Map<String, Integer> lichen_strainsUpdate;

    // every step
    public int factories_per_team;
}
