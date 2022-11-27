package com.luxai;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

/**
 * Hello world!
 *
 */
public class Bot
{
    public static boolean DEBUG = false;
    public static boolean REPLAY = false;

    public static void main( String[] args ) throws IOException {
        Agent agent = new Agent();
        Scanner scanner = new Scanner(System.in);
        if (!REPLAY) {     // Play
            while (true) {
                String json = scanner.nextLine();   // Read input
                agent.updateState(json);            // Update state

                if (DEBUG) {
                    String fileIn = agent.me() + "_in.txt";
                    FileWriter fw = new FileWriter(fileIn, true);
                    fw.write(json + "\n");
                    fw.close();
                }

                // - - - Main section - - -
                String jsonAction = null;
                if (agent.step <= agent.obs.board.factories_per_team + 1)
                    jsonAction = agent.early_setup();
                else {
                    jsonAction = agent.act();
                }
                if (jsonAction == null) {
                    ObjectMapper objectMapper = new ObjectMapper();
                    jsonAction = objectMapper.createObjectNode().toString();
                }
                // ^ ^ ^ Main section ^ ^ ^

                if (DEBUG) {
                    String fileOut = agent.me() + "_out.txt";
                    FileWriter fwOut = new FileWriter(fileOut, true);
                    fwOut.write(agent.step + ": " + jsonAction + "\n");
                    fwOut.close();
                }

                System.out.println(jsonAction);     // Output command
            }
        }
        else {  // Replay
            List<String> reverseList = Files.readAllLines(Paths.get("C:\\Users\\player_0_in.txt"), Charset.defaultCharset());
            for (String json : reverseList) {
                agent.updateState(json);            // Update state
                String jsonAction = null;
                if (agent.step <= agent.obs.board.factories_per_team + 1)
                    jsonAction = agent.early_setup();
                else {
                    jsonAction = agent.act();
                }
                if (jsonAction == null) {
                    ObjectMapper objectMapper = new ObjectMapper();
                    jsonAction = objectMapper.createObjectNode().toString();
                }
            }
        }
    }
}
