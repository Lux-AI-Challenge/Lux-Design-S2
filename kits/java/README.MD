# Lux AI Season 2 Java Kit

This is the folder for the Java kit. Please make sure to read the instructions as they are important regarding how you will write a bot and submit it to the competition.

Make sure to check our [Discord](https://discord.gg/aWJt3UAcgn) or the [Kaggle forums](https://www.kaggle.com/c/lux-ai-season2/discussion) for announcements if there are any breaking changes.

## Requirements

You will need Maven with Java 11 installed and Python >=3.7, <3.11 (which should come with the dependencies you installed for the environment).

## Getting Started

To get started, download this folder from this repository.

Your core agent code will go into Agent class, and you can create and use more files to help you as well. You should leave `main.py` files alone as that code enables your agent to compete against other agents locally and on Kaggle.

## Make fat jar with maven

```bash
mvn clean compile assembly:single
```

## Running local games

1. Create fat-jar with maven
2. Move jar to subfolder localrun
3. Run with main.py

```
luxai-s2 main.py main.py -o replay.json
```

## Compile for uploading to leaderboard evaluation

1. Create an archive with JavaBot.jar and main.py from root dir
    ```
    tar -cvzf submission.tar.gz main.py -C localrun JavaBot.jar
    ```
2. Submit to leaderboard server