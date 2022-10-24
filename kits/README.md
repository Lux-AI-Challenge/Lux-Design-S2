# Lux AI Season 2 Kits

This folder contains all official kits provided by the Lux AI team for the Lux AI Challenge Season 1.

In each starter kit folder we give you all the tools necessary to compete. Make sure to read the README document carefully. For debugging, you may log to standard error e.g. `console.error("hello")` or `print("hello", file=sys.stderr)`, this will be recorded and saved into a errorlogs folder for the match for each agent and will be recorded by the competition servers.

To run a episode with verbosity level 2 (higher is more verbose), and seed 42:

```
python luxai_runner/runner.py kits/python/main.py kits/js/main.js -s 42 -v 2
```

<!-- TODO: add instructions on watching replay either live or through visualizer -->
<!-- TODO: add instructions on gym interface -->

For submission to the kaggle challenge, you can first test your submission by running



## Kit Structure

Each agent is a folder of files with a `main.py` file and a language dependent `agent.py/agent.js/agent...` file. You can generally ignore `main.py` and focus on the `agent` file. For the rest of this document we will use a python based syntax and assume you are working with the python kit but the instructions apply to all kits.

In the `agent.py` file, we define a simple class that holds your agent, you can modify as you wish but be aware of the two functions given. By default, the kit calls the `early_setup` function for when the agent is bidding and placing down all its factories. It then calls the `act` function for the rest of the game. These functions have parameters `step` equal to the environment time step, `obs` equal to the actual observations, and `remainingOverageTime` representing how much extra time you have left before having to make a decision. See [here] for details on how bots are timed and limited.

These two functions are where you define your agent's logic for both the early phase and the actual game phase of Lux AI season 2. In all kits, example code has been provided to show how to read the observation and return an action to submit to the environment.

## Environment Actions

In each episode there are two competing teams, both of which control factories and units.

Factories have 3 possible actions, `build_light`, `build_heavy`, and `water`.

Units (light or heavy robots) have 5 possible actions: `move`, `dig`, `transfer`, `pickup`, `self_destruct`, `recharge`; where `move, dig, self_destruct` have power costs

In Lux AI Season 2, the unit's actual action space is a list of actions representing it's action queue and your agent will set this action queue to control units. This action queue max size is `env_cfg.UNIT_ACTION_QUEUE_SIZE`. Each turn, the unit executes the action at the front of the queue. If the action is marked as to be repeated, it is replaced to the back of the queue.

In code, actions can be given to units as so

```
actions[unit_id] = [action_0, action_1, ...]
```

Importantly, whenever you submit a new action queue, it incurs an additional power cost for communication of `env_cfg.UNIT_ACTION_QUEUE_POWER_COST` power to the unit. While you can still compete by submitting a action queue with a single action to every unit (like most environments and Lux AI Season 1), this is power inefficient and would be disadvantageous.

See the example code in the corresponding `agent.py` file for how to give actions, how to set them to repeat or not, and the various utility functions to validate if an action is possible or not (e.g. does the unit have enough power to perform an action).

## Environment Observations

First, the environment configuration being run is given to your agent. It will be stored under `self.env_cfg`, see the code for details on how to access for different languages.

The general observation given to your bot in the kits will look like below. `Array(n, m)` indicates an array with `n` rows and `m` columns. `[player_id]: {...}` indicates that `{...}` data can be under any player_id key, and the same logic follows for `[unit_id]: {...}`. Note that the gym interface returns just the "obs" key as the observation.

```
{
  "obs": {
    "units": {
      [player_id]: {
        [unit_id]: {
          "team_id": int,
          "unit_id": str,
          "power": int,
          "unit_type": "LIGHT" or "HEAVY",
          "pos": Array(2),
          "cargo": { "ice": int, "ore": int, "water": int, "metal": int },
          "action_queue": Array(N, 5)
        }
      }
    },
    "factories": {
      [player_id]: {
        [unit_id]: {
          "team_id": int,
          "unit_id": str,
          "power": int,
          "pos": Array(2),
          "cargo": { "ice": int, "ore": int, "water": int, "metal": int },
          "strain_id": int,
        }
      }
    },
    "board": {
      "rubble": Array(64, 64),
      "ice": Array(64, 64),
      "ore": Array(64, 64),
      "lichen": Array(64, 64),
      "lichen_strains": Array(64, 64),
      "spawns": Array(K, 2),
      "factories_per_team": int
    },
    "weather": Array(1000),
    "real_env_steps": int
  },
  "step": int,
  "remainingOverageTime": int,
  "player": str
}
```

Every kit has an `Agent` class that defines two functions, `early_setup` and `act` with parameters `step`, `obs` and `remainingOverageTime` corresponding to the values in the definition above. Note that for the `act` function, `obs["real_env_steps"]` is used instead. This subtracts the time spent bidding and placing factories in `early_setup` and so the first call to `act` will be with `step = 0` and `act` will be called `max_episode_length` times (default 1000).