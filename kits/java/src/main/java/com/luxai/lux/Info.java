package com.luxai.lux;

import com.fasterxml.jackson.annotation.JsonProperty;

public class Info {
    @JsonProperty("env_cfg")
    public Environment envConfig;
}
