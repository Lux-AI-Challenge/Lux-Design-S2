package com.luxai.lux.objectmapper;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.luxai.Agent;
import com.luxai.lux.State;

import java.util.Map;

public class Mapper {

    public static String getJson(Object object) throws JsonProcessingException {
        ObjectMapper objectMapper = new ObjectMapper();
        return objectMapper.writeValueAsString(object);
    }

    public static void updateState(Agent agent, String json) throws JsonProcessingException {
        ObjectMapper objectMapper = new ObjectMapper();
        // check first step
        if (agent.obs == null) {
            json = json
                    .replace("rubble", "rubble_init")
                    .replace("lichen_strains", "strains_init")
                    .replace("lichen", "lichen_init");

            State state = objectMapper.readValue(json, State.class);
            agent.obs = state.obs;
            agent.step = state.step;
            agent.remainingOverageTime = state.remainingOverageTime;
            agent.player = state.player;
            agent.env_cfg = state.info.env_cfg;
        }
        else {
            State state = objectMapper.readValue(json, State.class);
            agent.step = state.step;
            agent.remainingOverageTime = state.remainingOverageTime;
            agent.obs.teams = state.obs.teams;
            agent.obs.real_env_steps = state.obs.real_env_steps;
            agent.obs.units.clear(); agent.obs.units = state.obs.units;
            agent.obs.factories.clear(); agent.obs.factories = state.obs.factories;
            if (state.obs.real_env_steps < 0)
                agent.obs.board.valid_spawns_mask = state.obs.board.valid_spawns_mask;


            if (state.obs.board.rubbleUpdate != null) {
                for (Map.Entry<String, Integer> entry : state.obs.board.rubbleUpdate.entrySet()) {
                    String[] coordinates = entry.getKey().split(",");
                    int x = Integer.parseInt(coordinates[0]);
                    int y = Integer.parseInt(coordinates[1]);
                    agent.obs.board.rubble[x][y] = entry.getValue();
                }
            }

            if (state.obs.board.lichenUpdate != null) {
                for (Map.Entry<String, Integer> entry : state.obs.board.lichenUpdate.entrySet()) {
                    String[] coordinates = entry.getKey().split(",");
                    int x = Integer.parseInt(coordinates[0]);
                    int y = Integer.parseInt(coordinates[1]);
                    agent.obs.board.lichen[x][y] = entry.getValue();
                }
            }

            if (state.obs.board.lichen_strainsUpdate != null) {
                for (Map.Entry<String, Integer> entry : state.obs.board.lichen_strainsUpdate.entrySet()) {
                    String[] coordinates = entry.getKey().split(",");
                    int x = Integer.parseInt(coordinates[0]);
                    int y = Integer.parseInt(coordinates[1]);
                    agent.obs.board.lichen_strains[x][y] = entry.getValue();
                }
            }
        }
    }

}
