# Lux AI Challenge Season 2 Specifications - Detailed Version

This specifications document complements the original one at https://github.com/Lux-AI-Challenge/Lux-Design-S2/blob/main/specs.md but will cover the game engine line by line from [creating the environment](#environment-creation) to [receiving actions to returning observations](#environment-step---overview).

For some notation, `self` will always reference the [LuxAI_S2](https://github.com/Lux-AI-Challenge/Lux-Design-S2/blob/main/luxai_s2/luxai_s2/env.py#L70) environment object, and upper cased varibales refer to environment configuration parameters in [config.py](https://github.com/Lux-AI-Challenge/Lux-Design-S2/blob/main/luxai_s2/luxai_s2/config.py).

## Environment Creation

There are two ways to create the LuxAI environment, of which the recommended way is to do

```python
from luxai_s2 import LuxAI_S2
custom_env_cfg = dict()
env = LuxAI_S2(collect_stats=False, **custom_env_cfg)
env.reset()
```

where `collect_stats=True` will collect aggregate stats for an episode stored in `env.state.stats` and `custom_env_cfg` can be a custom env configuration to override the default. The custom env configuration may only replace existing keys as defined in [config.py](https://github.com/Lux-AI-Challenge/Lux-Design-S2/blob/main/luxai_s2/luxai_s2/config.py).

The other way to create an environment is to do

```
import luxai_s2
custom_env_cfg = dict()
env = gym.make("LuxAI_S2-v0", collect_stats=False, **custom_env_cfg)
env.reset()
```

Upon creation, an empty `State` object is created and the default agent names given are `"player_0", "player_1"`.

## Environment State

When an environment is created, it carries along with it a [State](https://github.com/Lux-AI-Challenge/Lux-Design-S2/blob/main/luxai_s2/luxai_s2/state/state.py) object, which contains every point of data necessary to recreate a state. This allows you to do the following to set and get states

```
import copy
state = env.get_state()
copy.deepcopy(state)

new_env = LuxAI_S2(collect_stats=False, **custom_env_cfg)
new_env.reset()
new_env.set_state(state)
# now new_env is the exact same as env
```

## Environment real_env_steps vs env_steps

There's an important distinction between `env.state.env_steps` and `env.state.real_env_steps`. `env_steps` counts from 0 and includes steps in the bidding and factory placement phases. `real_env_steps` starts negative, and becomes 0 when the normal game phase starts.

For RL practioners, what this means is this is very non-standard and if you wish to do RL over the entire game, use `env_steps` as the time. If you wish to do RL over just the normal game phase (recommended) use `real_env_steps` as the timer or track it yourself.

## Environment Step - Overview

Note: If you are interested in diving into the game logic and code, we recommend reading the rest of the sections with the [env.py](https://github.com/Lux-AI-Challenge/Lux-Design-S2/blob/main/luxai_s2/luxai_s2/env.py) file open.

This section gives a high level overview of the larger chunks of the code. Subsequent sections will go through some chunks more finely (and are linked in the bullet points below)

When both teams have submitted actions, we call `env.step(actions)` and enter the `env.step` function and do the following.

1. We check if `actions` is empty. If so, we raise a `ValueError` and end the game. On competition servers and the CLI tool this can never happen but if you are working with the environment directly this can

2. Check which stage of the game we are in (bidding phase, factory placement phase, or normal game phase). Depending on phase, we will handle it appropriately. The subsequent bullet points will discuss the normal game phase handling. [Game Phase Determination](#environment-step---game-phase-determination), [Bidding Phase](#environment-step---bidding-phase), [Factory Placement Phase](#environment-step---factory-placement-phase)

3. In the normal game phase, if `self.env_cfg.validate_action_space` is True, and then validate the actions using the current action_space of the game. This setting can be slow, but is always turned on in competition servers. [Validating Actions via Action Space](#environment-step---normal-phase-validating-actions-via-action-space)

4. For each unit with an action provided by both teams, we save all the factory / robot actions and raise a `ValueError` if there's a formatting error. For robots specifically, if a robot has `ACTION_QUEUE_POWER_COST` or more power, it is deducted from the robots power and its action queue is replaced with the new provided one. [Processing Actions and Updating Action Queues](#environment-step---normal-phase-processing-actions-and-updating-action-queues)

5. For each factory we take its latest action, and for each robot we take the action at the front of its action queue and put them all into one `dict` called `actions_by_type` which tracks all actions by the action type (e.g. dig, move etc.). [Tracking Actions by Type](#environment-step---normal-phase-tracking-actions-by-type)

6. We first handle digging actions (simultaneous). [Handling Dig Actions](#environment-step---normal-phase-handling-dig-actions)

7. We then handle self destruct actions (simultaneous). [Handling Self Destruct Actions](#environment-step---normal-phase-handling-self-destruct-actions)

8. We then handle factory building actions (simultaneous). [Handling Factory Build Actions](#environment-step---normal-phase-handling-factory-build-actions)

9. We then handle robot movement actions and resolve collisions (simultaneous). [Handling Robot Movement Actions](#environment-step---normal-phase-handling-robot-movement-actions)

10. We then handle recharge actions (simultaneous). [Handling Recharge Actions](#environment-step---normal-phase-handling-recharge-actions)

11. We then compute the water costs by finding the connected lichen tiles for each factory and handle factory watering actions (simultaneous). When a factory waters, 2 lichen is added for each new lichen and connected lichen tile. [Handling Water actions](#environment-step---normal-phase-handling-water-actions)

12. We then handle resource transfer actions (simultaneous). [Handling Resource Transfer Actions](#environment-step---normal-phase-handling-resource-transfer-actions)

13. Finally we handle resource pickup actions (in order of robot unit id). [Handling Resource Pickup Actions](#environment-step---normal-phase-handling-resource-pickup-actions)

14. 1 lichen is subtracted from each tile on the board and then we clip the value range to [0, `MAX_LICHEN_PER_TILE`]. Any tile with no lichen has the strain_id stored in `self.state.board.lichen_strains` set to -1.

15. Refine all resources and consume water for factories, destroying any factories without any water left to consume. Power is then added to every robot and factory. [Updating Resources, Water, and Power](#environment-step---normal-phase-updating-resources-water-and-power)

16. Compute rewards (equal to total owned lichen), update the `self.env_steps` variable, figure out which players are `done` playing (have lost or game is completed due to timelimit) and then return the same observation for both players using `self.state.get.obs()`. [Generating Observations](#environment-step---normal-phase-generating-observations)

The remaining sections will elaborate on some of the above steps run in the `env.step` function

## Environment Step - Game Phase Determination

If `self.state.env_steps == 0`, then we are in the bidding phase. Otherwise if `self.state.real_env_steps < 0` then we are in the factory placement period.

These two phases are described in the next two sections and are handled in `self._step_early_game(actions)`.

Note that `self.state.real_env_steps` is equal to `self.state.env_steps - (self.state.board.factories_per_team * 2 + 1)`, effectively removing the turns spent on bidding and factory placement by both teams so that 0 means the start of the normal phase.

## Environment Step - Bidding Phase
In the `self._step_early_game(actions)` call, we call `self._handle_bid(actions)` to handle this phase. We expect actions to be of the form `dict(player_0=..., player_1=...)` where the player actions are in this format `dict(bid: int, faction: str)`.

Moreover, in addition to handling bids, if one team/agent is "failed" that means we end the game and the other team wins.



As actions is a dictionary, we iterate over its keys (agent names) and actions. If action is None, we fail that agent for not submitting a bid action. If agent name is not in the predefined possible agents (player_0, player_1), we raise a ValueError.

If in the bid action there's an unrecognized faction name, (possible values are AlphaStrike, MotherMars, TheBuilders, FirstMars), then we fail the agent.

Otherwise, we think the bid action is formatted correctly and proceed to create a new [Team](https://github.com/Lux-AI-Challenge/Lux-Design-S2/blob/main/luxai_s2/luxai_s2/team.py) object stored in `self.state.teams[agent_name]`. We then give this team an initial pool of water and metal equal to `INIT_WATER_METAL_PER_FACTORY * self.state.board.factories_per_team` where `factories_per_team` is the number of factories both teams get to place and is randomly chosen each game.

We then check if the bid is valid. If the bid is greater than the initial pool of metal/water, then we fail the agent. 

We then keep track of which agent gave the highest bid, and mark which team gets to `place_first` and which does not.


## Environment Step - Factory Placement Phase
In the `self._step_early_game(actions)` call, we call `self._handle_factory_placement_step(actions)` to handle this phase. We expect actions to be of the form `dict(player_0=..., player_1=...)` where the player actions are in this format `dict(spawn: Array (2,), metal: int, water: int)`.

Note that this phase is simultaneous in that both teams can submit actions, however we will only process one of them based on whose turn it is to place a factory.

We first check which team (player_0 or player_1) is allowed to place a factory. The team that is allowed to place a factory keeps its action and we process it. We check for validity of the action by checking 1. if you have factories left to place, 2. if you gave non-negative water and metal spawn amounts, 3. if you have that much water and metal left in your initial pool to give to the new factory, and 4. if the given spawn location isn't too close to another or on top of resources. If any part fails, we skip the action and log a warning.

If valid, we then create a new factory at the desired spawn location with the desired water and metal and `INIT_POWER_PER_FACTORY` power. Then we subtract that water and metal from your team's initial pool and reduce `factories_to_place` by 1.


## Environment Step - Normal Phase: Validating Actions via Action Space

Once entering the normal phase, if `self.env_cfg.validate_action_space` is true, for each agent we use the action space of that agent to validate its action. This is somewhat slow and if you know your actions are always formatted correctly you can turn this off. But for competition servers this is always on. When validation is performed and the action is not valid, we log the error and reason and fail the agent that submitted that action.

The action space in the normal phase looks as so. We use [x] to represent a key with variable name x.

```
{
  [robot id]: 
    ActionsQueue(
      spaces.Box(
        low=np.array([0, 0, 0, 0, 0, 1]),
        high=np.array([5, 4, 4, config.max_transfer_amount + 1, 1, 9999]),
        shape=(6,),
        dtype=np.int64,
      ),
      config.UNIT_ACTION_QUEUE_SIZE
    )
  [factory id]: spaces.Discrete(3)
}
```
See the document on [spaces.Box/spaces.Discrete](https://gymnasium.farama.org/api/spaces/fundamental/) for how they work. See the [act_space.py](https://github.com/Lux-AI-Challenge/Lux-Design-S2/blob/main/luxai_s2/luxai_s2/spaces/act_space.py) file for details on how `ActionsQueue` works.

Actions don't need to include an action for every robot and factory, so we check for partial containment. 

## Environment Step - Normal Phase: Processing Actions and Updating Action Queues

For every factory (which has IDs prefixed with "factory"), we directly keep the submitted action and store it.

For every robot (which has IDs prefixed with "unit"), we find its robot type via `unit.unit_type.name` and subsequently find its `ACTION_QUEUE_POWER_COST`. If the robot doesn't have enough power, we don't update the action queue.

We then check if the team's submitted action queue for this robot is formatted correctly (check if it's a list and it has 2 dimensions). If not, we fail that robot's team.

If the new action queue is correct and the robot has enough power to update, we use the new action queue given for the robot (the action space for this is defined in [act_space.py](https://github.com/Lux-AI-Challenge/Lux-Design-S2/blob/main/luxai_s2/luxai_s2/spaces/act_space.py), see comments to understand parameterization) and truncate it to length `UNIT_ACTION_QUEUE_SIZE` and replace the old action queue. We then subtract `ACTION_QUEUE_POWER_COST` from this robot's power.

## Environment Step - Normal Phase: Tracking Actions by Type

For each agent, if it was failed in a previous step we skip it. For functioning agents, we retrieve the action at the front of each owned unit's action queue, as well as each action from each factory if there is one. We then store them in `actions_by_type` which maps action type (e.g. Dig, Build) to a list of all actions from all agents of that type.

## Environment Step - Normal Phase: Handling Dig Actions
## Environment Step - Normal Phase: Handling Self Destruct Actions
## Environment Step - Normal Phase: Handling Factory Build Actions
## Environment Step - Normal Phase: Handling Robot Movement Actions
## Environment Step - Normal Phase: Handling Recharge Actions

## Environment Step - Normal Phase: Handling Water Actions

## Environment Step - Normal Phase: Handling Resource Transfer Actions

## Environment Step - Normal Phase: Handling Resource Pickup Actions

## Environment Step - Normal Phase: Updating Resources Water, and Power

## Environment Step - Normal Phase: Generating Observations