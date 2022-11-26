package com.luxai.lux;

import java.util.Map;

public class Obs {
    public Map<String, Map<String, Robot>> units;
    public Map<String,Team> teams;
    public Map<String,Map<String, Factory>> factories;
    public Board board;
    public int[] weather_schedule;
    public int real_env_steps;
}
