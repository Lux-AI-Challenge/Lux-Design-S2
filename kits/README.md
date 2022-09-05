# Lux AI Season 2 Kits

This folder contains all official kits provided by the Lux AI team for the Lux AI Challenge Season 1.

In each starter kit folder we give you all the tools necessary to compete. Make sure to read the README document carefully. For debugging, you may log to standard error e.g. `console.error("hello")` or `print("hello", file=sys.stderr)`, this will be recorded and saved into a errorlogs folder for the match for each agent and will be recorded by the competition servers.

To run a episode

```
python luxai_runner/runner.py kits/python/main.py kits/js/main.js -v 2
```

## Observations Definition

Season 2 is split into two phases. The Early Phase and actual Game Phase.

During the Early Phase, your bot must bid for a new factory and select a faction to represent. Then y


```
{
  "obs": {
    "rubble": Array(64, 64),
    "ice": Array(64, 64),
    "ore": Array(64, 64),
    "lichen": Array(64, 64),
    "lichen_strains": Array(64, 64),
    "spawns": Array(K, 2)
  },
  "step": int,
  "remainingOverageTime": int,
  

}
```