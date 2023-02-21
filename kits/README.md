# Lux AI Season 2 Kits

This folder contains all kits for the Lux AI Challenge Season 2. It covers the [Kit Structure](#kit-structure), [Forward Simulation](#forward-simulation), Envionment [Actions](#environment-actions) and [Observations](#environment-observations), as well as the general [Kit API](#kit-api). For those interested in the [RL starter kits/baselines](https://github.com/Lux-AI-Challenge/Lux-Design-S2/tree/main/kits/rl), we highly recommend reading those respective docs as they don't use the standard Kit API. For those who need to know what python packages are available on the competition server, see [this](https://github.com/Lux-AI-Challenge/Lux-Design-S2/tree/main/kits/available_packages.txt)

In each starter kit folder we give you all the tools necessary to compete. Make sure to read the README document carefully. For debugging, you may log to standard error e.g. `console.error("hello")` or `print("hello", file=sys.stderr)`, and will be recorded by the competition servers.

To run a episode with verbosity level 2 (higher is more verbose), seed 42, and save a replay to replay.json:

```
luxai-s2 kits/python/main.py kits/python/main.py -s 42 -v 2 -o replay.json
```

To then watch the replay, upload replay.json to http://s2vis.lux-ai.org/

Alternatively you can generate a openable HTML file to watch it as well by specifying the output as .html

```
luxai-s2 kits/python/main.py kits/python/main.py -s 42 -v 2 -o replay.html
```

For an in-depth tutorial detailing how to start writing an agent, there is a [online Jupyter Notebook](https://www.kaggle.com/code/stonet2000/lux-ai-challenge-season-2-tutorial-python) that you can follow (only in python). We highly recommend at least skimming over this as season 2 has some specific quirks that make it different than your standard AI gym environments. Specifically they affect the [environment actions](#environment-actions) mostly.

## Kit Structure

Each agent is a folder of files with a `main.py` file and a language dependent `agent.py/agent.js/agent...` file. You can generally ignore `main.py` and focus on the `agent` file which is where you write your logic. For the rest of this document we will use a python based syntax and assume you are working with the python kit but the instructions apply to all kits.

In the `agent.py` file, we define a simple class that holds your agent, you can modify as you wish but be aware of the two functions given. By default, the kit calls the `early_setup` function for when the agent is bidding and placing down all its factories. It then calls the `act` function for the rest of the game. These functions have parameters `step` equal to the environment time step, `obs` equal to the actual observations, and `remainingOverageTime` representing how much extra time you have left before having to make a decision.

These two functions are where you define your agent's logic for both the early phase and the actual game phase of Lux AI season 2. In all kits, example code has been provided to show how to read the observation and return an action to submit to the environment.

## Forward Simulation

For certain strategies it helps to know where units are and what the lichen is like after a few steps.

For the JS kit, forward simulation is possible by setting the `FORWARD_SIM` value in main.py. For the python kit you can simply use the `lux.forward_sim` tool in the `lux` kit folder. In all kits that have forward simulation enabled they will return a list of observations representing the current and next few observations.

## Environment Actions

In each episode there are two competing teams, both of which control factories and units.

In the early phase, the action space is different than the normal game phase. See the starter kit codes (agent.py file) for how they are different.

During the normal game phase, factories have 3 possible actions, `build_light`, `build_heavy`, and `water`. Units/Robots (light or heavy robots) have 5 possible actions: `move`, `dig`, `transfer`, `pickup`, `self_destruct`, `recharge`; where `move, dig, self_destruct` have power costs

In Lux AI Season 2, the robots's actual action space is a list of actions representing it's action queue and your agent will set this action queue to control robots. This action queue max size is `env_cfg.UNIT_ACTION_QUEUE_SIZE`. Each turn, the unit executes the action at the front of the queue, and repeatedly executes this a user-specified `n` times. Moreover, action execution counts towards `n` only when it is succesful, so if your robot runs out of power or a resource to transfer, it won't be counted towards `n`. Finally, each action can specify a `repeat` value. If `repeat == 0` then after `n` executions the action is removed. If `repeat > 0`, then the action is recycled to the back of the queue and sets `n = repeat` insead of removing the action.

In code, actions can be given to units as so

```
actions[unit_id] = [action_0, action_1, ...]
```

Importantly, whenever you submit a new action queue, the unit incurs an additional power cost to update the queue of `env_cfg.ROBOTS[<robot_type>].ACTION_QUEUE_POWER_COST` power. While you can still compete by submitting a action queue with a single action to every unit (like most environments and Lux AI Season 1), this is power inefficient and would be disadvantageous. Lights consume 1 power and Heavies consume 10 power to update their action queue,

See the example code in the corresponding `agent.py` file for how to give actions, how to set their `n` and `repeat` values to control execution count and recycling, and the various utility functions to validate if an action is possible or not (e.g. does the unit have enough power to perform an action). For those interested in **how the exact `action_i` vector is encoded, see [this section on our advanced specs document](https://github.com/Lux-AI-Challenge/Lux-Design-S2/blob/main/docs/advanced_specs.md#action-vector-encoding).**

## Environment Observations

First, the environment configuration being run is given to your agent. It will be stored under `self.env_cfg`, see the code for details on how to access for different languages.

The general observation given to your bot in the kits will look like below. `Array(n, m)` indicates an array with `n` rows and `m` columns. `[player_id]: {...}` indicates that `{...}` data can be under any player_id key, and the same logic follows for `[unit_id]: {...}`. Note that the gym interface returns just the "obs" key as the observation. For a concrete version of the env configurations and observations with numbers, see `sample_obs.json` and `sample_env_cfg.json` in this folder.

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
          "action_queue": Array(N, 6)
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
      "rubble": Array(48, 48),
      "ice": Array(48, 48),
      "ore": Array(48, 48),
      "lichen": Array(48, 48),
      "lichen_strains": Array(48, 48),
      "valid_spawns_mask": Array(48, 48),
      "factories_per_team": int
    },
    "teams": {
      [player_id]: {
        "team_id": int,
        "faction": str,
        "water": int,
        "metal": int,
        "factories_to_place": int,
        "factory_strains": Array<int>,
        "place_first": bool,
        "bid": int
      }
    },
    "real_env_steps": int, # note this can be negative due to there being two phases of gameplay
    "global_id": int # only used for reconstructing a LuxAI_S2 State object
  },
  "step": int,
  "remainingOverageTime": int, # total amount of time your bot can use whenever it exceeds 2s in a turn
  "player": str # your player id
}
```

Every kit has an `Agent` class that defines two functions, `early_setup` and `act` with parameters `step`, `obs` and `remainingOverageTime` corresponding to the values in the definition above. Note that for the `act` function, `obs["real_env_steps"]` is used instead. This subtracts the time spent bidding and placing factories in `early_setup` and so the first call to `act` will be with `step = 0` and `act` will be called `max_episode_length` times (default 1000).

## Kit API

All kits come with a interactable API to get data about the current state/observation of the environment. For specific details of how to use it you should refer to the code/docs in the respective kit folders. The game state or formatted observation looks as so
```python
class GameState:
    env_steps: int # number of env steps passed
    env_cfg: dict # current env configuration
    board: Board # the game board
    units: Dict[str, Dict[str, Unit]] # maps agent ID (player_0, player_1) to a dictionary mapping unit ID to unit objects
    factories: Dict[str, Dict[str, Factory]] # maps agent ID (player_0, player_1) to a dictionary mapping unit ID to factory objects
    teams: Dict[str, Team] # maps agent ID (player_0, player_1) to a Team object
```

The board object looks as so

```python
class Board:
    rubble: Array
    ice: Array
    ore: Array
    lichen: Array
    lichen_strains: Array # the id of the lichen planted at each tile, corresponds with factory.strain_id
    factory_occupancy_map: Array # -1 everywhere. Otherwise has the numerical ID of the factory (equivalent to factory.strain_id) that occupies that tile
    factories_per_team: int # number of factories each team gets to place initially
    valid_spawns_mask: Array # A mask array of the map with 1s where you can spawn a factory and 0s where you can't
```

Each `Unit` object comes with functions to generate the action vector for actions like move and dig, as well as cost functions that return the power cost to perform some actions.

Each `Factory` object comes with functions to generate actions as well as compute the cost of building robots and watering lichen.

Finally, the `Team` object holds the initial pool of water and metal the team has during the early phase, the number of factories left to place, and the strain ids of the owned factories.
