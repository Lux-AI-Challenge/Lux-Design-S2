package com.luxai;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.luxai.objectmapper.Mapper;

import java.io.IOException;
import java.util.Scanner;

/**
 * Main class
 * Lux-ai-2022
 */
public class Bot
{
    public static void main( String[] args ) throws IOException {
        Agent agent = new Agent();
        Scanner scanner = new Scanner(System.in);
        while (true) {
            String jsonIn = scanner.nextLine();             // Read input
            String jsonOut = processing(agent, jsonIn);     // Main function
            System.out.println(jsonOut);                    // Output command
        }
    }

    public static String processing(Agent agent, String json) throws JsonProcessingException {
        Mapper.updateState(agent, json);            // Update state
        String jsonAction = null;
        if (agent.obs.realEnvSteps < 0)
            jsonAction = agent.earlySetup();
        else {
            jsonAction = agent.act();
        }
        if (jsonAction == null) {
            ObjectMapper objectMapper = new ObjectMapper();
            jsonAction = objectMapper.createObjectNode().toString();
        }
        return jsonAction;
    }
}
