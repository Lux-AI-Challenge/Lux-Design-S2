# Lux AI Season 2 CLI Tool

To run a match between two agents, run


```
luxai2022 path/to/main.py path/to_another/main.py -o replay.json
```

For additional help run `luxai2022 --help`

To run a tournament style leaderboard, run 

```
python -m luxai_runner.cli \
  path/to/bot1/main.py path/to/bot2/main.py \
  path/to/bot3/main.py path/to/bot4/main.py \
  -o replays/replay.json --tournament -v 0
```

or specify a folder where each sub-folder contains a main.py file e.g.

```
python -m luxai_runner.cli path/to/ -o replays/replay.json --tournament -v 0
```

which will find agents `path/to/bot1/main.py`, `path/to/bot2/main.py` etc.

This will live print a running leaderboard like below, showing the bot/player, the ELO rating, and number of episodes its been in. At the moment it only computes an ELO rating (with K factor 32) and does random matchmaking. All replays are saved to `replays/replay_<episode_id>.json` as specified to the `-o` argument in the script above.

```
==== luxai2022_tourney ====
Player                              | Rating  | Episodes      
--------------------------------------------------------------
path/to/bot1/main.py_2               | 1091.490| 22            
path/to/bot1/main.py_0               | 1055.307| 17            
path/to/bot1/main.py_3               | 1043.040| 13            
path/to/bot1/main.py_1               | 800.6280| 22            
--------------------------------------------------------------
2 episodes are running
```
