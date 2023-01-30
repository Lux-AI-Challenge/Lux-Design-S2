# Lux AI Season 2 CLI Tool

To run a match between two agents, run

```
luxai-s2 path/to/main.py path/to_another/main.py -o replay.json
```

which saves a replay file to `replay.json` which can be watched at https://s2vis.lux-ai.org. You can also specify `-o replay.html` which lets you open the visualizer directly and auto load the replay.

For additional help run `luxai-s2 --help`

To run a tournament style leaderboard with all kinds of agents, run 

```
luxai-s2 \
  path/to/bot1/main.py path/to/bot2/main.py \
  path/to/bot3/main.py path/to/bot4_cpp/build/agent.out \
  -o replays/replay.json --tournament -v 0 --tournament_cfg.concurrent=2
```

or specify a folder where each sub-folder contains a main.py file e.g.

```
luxai-s2 path/to/ \
  -o replays/replay.json --tournament -v 0 --tournament_cfg.concurrent=2
```

which will find agents `path/to/bot1/main.py`, `path/to/bot2/main.py` etc. Note that this feature currently only works for python agents so it won't find the C++ agent for example. The `-v 0` turns off logging (as multiple matches are being run). `--tournament_cfg.concurrent=2` specifies to run at most two episodes concurrently, this can be increased to speed up total evaluation speed.

The above scripts will live print a running leaderboard like below, showing the bot/player, their score/rating, and number of episodes its been in. 

ELO rating (with K factor 32) and Win/Tie/Loss systems are support and matchmaking is random. All replays are saved to `replays/replay_<episode_id>.json` as specified to the `-o` argument in the script above.

```
==== luxai_s2_tourney ====
Player                              | Rating  | Episodes      
--------------------------------------------------------------
path/to/bot1/main.py_2               | 1091.490| 22            
path/to/bot1/main.py_0               | 1055.307| 17            
path/to/bot1/main.py_3               | 1043.040| 13            
path/to/bot1/main.py_1               | 800.6280| 22            
--------------------------------------------------------------
2 episodes are running
```
