package com.luxai.utils;

import com.luxai.Agent;
import com.luxai.Bot;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;

/**
 * Use instead Bot for local debugging
 * Lux-ai-2022
 */
public class BotDebugger
{
    private static final boolean REPLAY = false;
    // true - read inputJsonFile via IDE
    // false - compile fat-jar, invoke local via luxai2022 and get inputJsonFile
    private static final String inputJsonFile = "C:\\Users\\player_0_in.txt";

    public static void main( String[] args ) throws IOException {
        Agent agent = new Agent();
        Scanner scanner = new Scanner(System.in);
        if (REPLAY) {     // Debug with input json log
            List<String> reverseList = Files.readAllLines(Paths.get(inputJsonFile), Charset.defaultCharset());
            for (String jsonIn : reverseList) {
                Bot.processing(agent, jsonIn);
            }
        }
        else {
            while (true) {
                String jsonIn = scanner.nextLine();                 // Read input
                String jsonOut = Bot.processing(agent, jsonIn);     // Main function

                String fileIn = agent.player + "_in.txt";           // Log input JSON to file
                FileWriter fw = new FileWriter(fileIn, true);
                fw.write(jsonIn + "\n");
                fw.close();

                String fileOut = agent.player + "_out.txt";         // Log output JSON to file
                FileWriter fwOut = new FileWriter(fileOut, true);
                fwOut.write(agent.step + ": " + jsonOut + "\n");
                fwOut.close();

                System.out.println(jsonOut);                        // Output command
            }
        }
    }
}
