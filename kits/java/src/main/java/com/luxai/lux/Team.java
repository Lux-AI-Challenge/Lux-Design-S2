package com.luxai.lux;

import com.fasterxml.jackson.annotation.JsonProperty;

public class Team {
    @JsonProperty("team_id")
    public int teamId;
    public String faction;
    public int water;
    public int metal;
    @JsonProperty("factories_to_place")
    public int factoriesToPlace;
    @JsonProperty("factory_strains")
    public int[] factoryStrains;
    @JsonProperty("place_first")
    public boolean placeFirst;
}
