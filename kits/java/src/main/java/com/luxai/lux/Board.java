package com.luxai.lux;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.ArrayList;
import java.util.Map;

public class Board {
    // step = 0
    @JsonProperty("rubble_init")
    public int[][] rubble;          // Mixed up X and Y. I don`t know it is my bad or json-feature
    public int[][] ore;             // Mixed up X and Y. I don`t know it is my bad or json-feature
    public int[][] ice;             // Mixed up X and Y. I don`t know it is my bad or json-feature
    @JsonProperty("lichen_init")
    public int[][] lichen;          // Mixed up X and Y. I don`t know it is my bad or json-feature
    @JsonProperty("strains_init")
    public int[][] lichen_strains;  // Mixed up X and Y. I don`t know it is my bad or json-feature
    public boolean[][] valid_spawns_mask;
    //public Map<String, ArrayList<ArrayList<Integer>>> spawns;

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
