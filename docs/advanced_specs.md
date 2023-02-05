# Lux AI Challenge Season 2 Specifications - Detailed Version

This specifications document complements the original one at https://github.com/Lux-AI-Challenge/Lux-Design-S2/blob/main/specs.md but will cover the game engine line by line from [creating the environment](#environment-creation) to [receiving actions to returning observations](#environment-step---overview).

For some notation, `self` will always reference the [LuxAI_S2](https://github.com/Lux-AI-Challenge/Lux-Design-S2/blob/main/luxai_s2/luxai_s2/env.py#L70) environment object, and upper cased variables refer to environment configuration parameters in [the EnvConfig object in config.py](https://github.com/Lux-AI-Challenge/Lux-Design-S2/blob/main/luxai_s2/luxai_s2/config.py).

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

For RL practitioners, what this means is this is very non-standard and if you wish to do RL over the entire game, use `env_steps` as the time. If you wish to do RL over just the normal game phase (recommended) use `real_env_steps` as the timer or track it yourself.

## Environment Step - Overview

Note: If you are interested in diving into the game logic and code, we recommend reading the rest of the sections with the [env.py](https://github.com/Lux-AI-Challenge/Lux-Design-S2/blob/main/luxai_s2/luxai_s2/env.py) file open.

This section gives a high level overview of the larger chunks of the code. Subsequent sections will go through some chunks more finely (and are linked in the bullet points below)

When both teams have submitted actions, we call `env.step(actions)` and enter the `env.step` function and do the following.

1. We check if `actions` is empty. If so, we raise a `ValueError` and end the game. On competition servers and the CLI tool this can never happen but if you are working with the environment directly this can

2. Check which stage of the game we are in (bidding phase, factory placement phase, or normal game phase). Depending on phase, we will handle it appropriately. The subsequent bullet points will discuss the normal game phase handling. [Game Phase Determination](#environment-step---game-phase-determination), [Bidding Phase](#environment-step---bidding-phase), [Factory Placement Phase](#environment-step---factory-placement-phase)

3. In the normal game phase, if `self.env_cfg.validate_action_space` is True, and then validate the actions using the current action_space of the game. This setting can be slow, but is always turned on in competition servers. [Validating Actions via Action Space](#environment-step---normal-phase-validating-actions-via-action-space)

4. For each unit with an action provided by both teams, we save all the factory / robot actions and raise a `ValueError` if there's a formatting error. For robots specifically, if a robot has `ACTION_QUEUE_POWER_COST` or more power, it is deducted from the robots power and its action queue is replaced with the new provided one. [Processing Actions and Updating Action Queues](#environment-step---normal-phase-processing-actions-and-updating-action-queues)

5. For each factory we take its latest action, and for each robot we take the action at the front of its action queue and put them all into one `dict` called `actions_by_type` which tracks all actions by the action type (e.g. dig, move etc.). Then these actions are validated against the current state [Tracking Actions by Type and Validating Them](#environment-step---normal-phase-tracking-actions-by-type-and-validating-them)

6. We first handle digging actions (simultaneous). [Handling Dig Actions](#environment-step---normal-phase-handling-dig-actions)

7. We then handle self destruct actions (simultaneous). [Handling Self Destruct Actions](#environment-step---normal-phase-handling-self-destruct-actions)

8. We then handle factory building actions (simultaneous). [Handling Factory Build Actions](#environment-step---normal-phase-handling-factory-build-actions)

9. We then handle robot movement actions and resolve collisions (simultaneous). [Handling Robot Movement Actions](#environment-step---normal-phase-handling-robot-movement-actions)

10. We then handle recharge actions (simultaneous). [Handling Recharge Actions](#environment-step---normal-phase-handling-recharge-actions)

11. We then compute the water costs by finding the connected lichen tiles for each factory and handle factory watering actions (simultaneous). When a factory waters, 2 lichen is added for each new lichen and connected lichen tile. [Handling Water actions](#environment-step---normal-phase-handling-water-actions)

12. We then handle resource transfer actions (simultaneous). [Handling Resource Transfer Actions](#environment-step---normal-phase-handling-resource-transfer-actions)

13. Finally we handle resource pickup actions (in order of robot unit id). [Handling Resource Pickup Actions](#environment-step---normal-phase-handling-resource-pickup-actions)

14. Note that for each robot action we also may repeat it or recycle it. [Action Repeat and Recycle](#environment-step---normal-phase-action-repeat-and-recycle)

15. 1 lichen is subtracted from each tile on the board and then we clip the value range to [0, `MAX_LICHEN_PER_TILE`]. Any tile with no lichen has the strain_id stored in `self.state.board.lichen_strains` set to -1.

16.. Refine all resources and consume water for factories, destroying any factories without any water left to consume. Power is then added to every robot and factory. [Updating Resources, Water, and Power](#environment-step---normal-phase-updating-resources-water-and-power)

17. Compute rewards (equal to total owned lichen), update the `self.env_steps` variable, figure out which players are `done` playing (have lost or game is completed due to timelimit) and then return the same observation for both players using `self.state.get.obs()`. [Generating Observations](#environment-step---normal-phase-generating-observations)

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
        high=np.array([5, 4, 4, config.max_transfer_amount + 1, 9999, 9999]),
        shape=(6,),
        dtype=np.int64,
      ),
      config.UNIT_ACTION_QUEUE_SIZE
    ),
  [factory id]: spaces.Discrete(3)
}
```
See the document on [spaces.Box/spaces.Discrete](https://gymnasium.farama.org/api/spaces/fundamental/) for how they work. See the [act_space.py](https://github.com/Lux-AI-Challenge/Lux-Design-S2/blob/main/luxai_s2/luxai_s2/spaces/act_space.py) file for details on how `ActionsQueue` works. Moreover note that **there can be a maximum of `config.UNIT_ACTION_QUEUE_SIZE` actions in an action queue.**

Actions don't need to include an action for every robot and factory, so we check for partial containment. Importantly not submitting an action for a robot leaves their current action queue unchanged.

### Action Vector Encoding

This section details how each action in the action queue is encoded. If you plan to use the rule-based starter kits there you can skip this as this is useful for those who plan to read/write the actions directly instead of through an API. Moreover note that there can be a maximum of `config.UNIT_ACTION_QUEUE_SIZE` actions in an action queue.

Let the action be variable `a`. In general an action vector has 6 dimensions so `len(a) == 6`.

`a[0]` encodes the type of action. 0 = move, 1 = transfer `X` amount of `R`, 2 = pickup `X` amount of `R`, 3 = dig, 4 = self destruct, 5 = recharge `X`. `X` represents an amount variable and `R` represents a resource encoding which are encoded in `a[3]` and `a[2]` respectively.

`a[1]` encodes the direction. 0 = center, 1 = up, 2 = right, 3 = down, 4 = left. This value is only used for move and transfer and is ignored for other actions. Note that moving center is like a `no-op` action and can be used to add timed delays that cost no power. However updating an action queue with move centers will still incur action queue update costs

`a[2] = R`, which represents the resource type (0 = ice, 1 = ore, 2 = water, 3 = metal, 4 power). This value is only used for transfer/pickup actions and is ignored for other actions.

`a[3] = X`, which represents the amount of resources transferred or picked up. This value is only used for transfer/pickup actions and is ignored for other actions.

`a[4]` encodes the `repeat` value. Once an action is exhausted we recycle the action to the back of the queue with `n = a[4] = repeat` if `repeat > 0`. Note that `repeat` must be between 0 and 9999 inclusive.

`a[5]`encodes the `n` value, encoding the number of times left to execute this action. `n` only decrements when the action is succesfully executed (e.g. had enough power to run). When `n = 1` and is decremented we consider recycling the action based on `a[4] = repeat`. Note that `n` must be between 1 and 9999 inclusive.

## Environment Step - Normal Phase: Processing Actions and Updating Action Queues

For every factory (which has IDs prefixed with "factory"), we directly keep the submitted action and store it.

For every robot (which has IDs prefixed with "unit"), we find its robot type via `unit.unit_type.name` and subsequently find its `ACTION_QUEUE_POWER_COST`. If the robot doesn't have enough power, we don't update the action queue.

We then check if the team's submitted action queue for this robot is formatted correctly (check if it's a list and it has 2 dimensions). If not, we fail that robot's team.

If the new action queue is correct and the robot has enough power to update, we use the new action queue given for the robot (the action space for this is defined in [act_space.py](https://github.com/Lux-AI-Challenge/Lux-Design-S2/blob/main/luxai_s2/luxai_s2/spaces/act_space.py), see comments to understand parameterization) and truncate it to length `UNIT_ACTION_QUEUE_SIZE` and replace the old action queue. We then subtract `ACTION_QUEUE_POWER_COST` from this robot's power.

## Environment Step - Normal Phase: Action Repeat and Recycle

For each action a robot has, it keeps track of two values, it's execution count `n`, and the `repeat` value which is the value of `n` to reset to when recycling the action. In code, we call the `unit.repeat_action(action)` function to handle this after processing each validated action.

For example, lets say there is a move up action with `n = 2` and `repeat=3`. The unit will try to move up twice. Each time the unit moves up successfully, we decrement `n` by 1. If `n` is decremented to `0`, we check `repeat`. 

If `repeat == 0`, then the action is removed from the action queue.

Else if `repeat > 0`, then we set `n = repeat` and recycle the action to the back of the action queue.

So once the unit has moved up twice, that move up action now has `n=3, repeat=3` and is at the end of the action queue. Once it comes to the front of the action queue, we will try to move up 3 times now, and once completed, we recycle the action with `n == repeat == 3`.

## Environment Step - Normal Phase: Tracking Actions by Type and Validating Them

For each agent, if it was failed in a previous step we skip it. For functioning agents, we retrieve the action at the front of each owned robot's action queue (without removing from the queue), as well as each action from each factory if there is one. We then store them in `actions_by_type` which maps action type (e.g. Dig, Build) to a list of all actions from all agents of that type.

We then use [validate_actions](https://github.com/Lux-AI-Challenge/Lux-Design-S2/blob/main/luxai_s2/luxai_s2/actions.py) function to go through every action in `actions_by_type` and figure out if against current game state at the start of the turn if the action would work. If we determine that an action would not be valid to execute in the current game state, we invalidate it and remove it from `actions_by_type`.

For dig, self destruct, movement, and factory building we check if the unit has enough power to do so. to perform the desired action, and if the action is formatted correctly.

For movement, we further check if it's moving onto an enemy factory or off the map and invalidate if so. Note that special to movement, power costs for movement depends on the amount of rubble on the target tile, so this is computed as well.

For resource transfer and pickup, we validate that you transfer to a location on the map and pickup from a friendly factory, otherwise we invalidate it.

For factory build actions, we further check if the factory has enough metal.

For factory water actions, we check if the factory has enough water.

**Note that if an action is invalidated, it doesn't mean the action is removed from the action queue.** The environment will execute actions on a "wait-until-success" model where **we only decrement an action's execution count if it passed this validation section.**

## Environment Step - Normal Phase: Handling Dig Actions

These are handled with `self._handle_dig_actions(actions_by_type)`. 

For each action in `actions_by_type["dig"]` we check if there is rubble on the tile under the unit digging. If so, we remove `DIG_RUBBLE_REMOVED` rubble  from the tile.

Else if there is > 0 lichen there, we remove `DIG_LICHEN_REMOVED` lichen. If after this the lichen left is 0, we add `DIG_RUBBLE_REMOVED` onto the tile.

Else if there is ice or ore on that tile (and at this point no rubble or lichen), we give the unit `DIG_RESOURCE_GAIN` units of ice or ore.

Finally we subtract `DIG_COST` from the unit's power

## Environment Step - Normal Phase: Handling Self Destruct Actions

These are handled with `self._handle_self_destruct_actions(actions_by_type)`. 

For each unit with this action, we remove it from the game and add `RUBBLE_AFTER_DESTRUCTION` rubble on the tile the unit was previously at. This also removes all lichen and the lichen strain on that tile.

## Environment Step - Normal Phase: Handling Factory Build Actions

These are handled with `self._handle_factory_build_actions(actions_by_type)`. 

For each factory with this action, we check first which unit the factory is building (HEAVY or LIGHT).

We then check if the factory still has enough metal and power to build that unit, and then remove that amount if it does and build that unit. The built unit then spawns on the center of the Factory with power initialized to how much power was spent.

The built unit is "dormant" in the turn it was built, meaning it will not do anything and can only start doing actions the next turn (so you can't predict the unit's id ahead of time and make it move right when it gets built).

## Environment Step - Normal Phase: Handling Robot Movement Actions

These are handled with `self._handle_movement_actions(actions_by_type)`

This primarily moves robots around and resolves collisions (see https://www.lux-ai.org/specs-s2#robots for the collision rules)

To do so, we keep track of each board tile that has units moving onto that tile with `heavy_entered_pos` and `light_entered_pos`. We also define `new_units_map` which maps x, y position hashes to a list of units.

The `self.state.board.units_map` keeps track of a map from hashes of x,y positions to the list of robots at that tile. Note that at the start of each turn it is guaranteed there is only one unit on each tile by game rules and code.

Now, for each robot and their move action, we skip it if its a move center action (a no-op). If not, we compute the power cost to move and subtract it from the unit. We then add the unit to the list of units entering the tile it moves onto (either `heavy_entered_pos` or `light_entered_pos` depending on unit type) as well as `new_units_map`. We further remove this unit from `self.state.board.units_map`. Thus after this part, `self.state.board.units_map` contains all the stationary units.


We iterate over all stationary units and add them to `new_units_map`. We also initialize the same type of dictionary called `new_units_map_after_collision`.

For each list of units in `new_units_map`, we do the following

If there is more than one unit, but there is also more than one heavy unit entering that tile (checking `heavy_entered_pos`), then all units are colliding and we find the two heavy units with the most power. The most powered heavy unit survives and loses power equal to half that of the 2nd most powered heavy unit. Similarly, if there is more than one light unit entering that tile and no heavy units entering or there to begin with, we apply the same process. All units that don't survive are destroyed.

If there is just one heavy unit entering a tile or just one stationary heavy unit on that tile and no other heavy units entering, then all other units (which are lights) are destroyed.

Any surviving unit is kept in `new_units_map_after_collision`, and we replace `self.state.board.units_map` with `new_units_map_after_collision`.


## Environment Step - Normal Phase: Handling Recharge Actions

These are handled with `self._handle_recharge_actions(actions_by_type)`

If unit has less power than what its recharge action desires, we do nothing. Otherwise, we call `unit.repeat_action`

## Environment Step - Normal Phase: Handling Water Actions

These are handled with `self._handle_factory_water_actions(actions_by_type)`

We first check the water cost of the factory. If the factory doesn't have enough water we log a warning and skip the action. For details on the algorithm that computes water cost see [this](#environment-water-cost-computation)

Otherwise we subtract the water cost from the factory's cargo and for each position in `factory.grow_lichen_positions` computed using `factory.cache_water_info` we increase lichen by 2 there and set the lichen strain there equal to the factory's `num_id` (also known as `strain_id`).

## Environment Step - Normal Phase: Handling Resource Transfer Actions

These are handled with `self._handle_transfer_actions(actions_by_type)`

This is simultaneous and generally any resources that overflow are wasted. Power and cargo (water, ice, metal, ore) can be transferred. In code and raw action vectors, `0 = ice, 1 = ore, 2 = water, 3 = metal, 4 power`.

We first iterate over each transfer action and determine how much each robot can actually transfer out. We permit transfer actions that specify transferring more than is possible, but we will only remove what is available of that resource from the unit's cargo.

Then for each transfer action and the actually transferable `transfer_amount`, we first find the position on the map the transfer action transfers to by adding the transfer direction to the transferring robot's position. 

If there is a factory there we transfer `transfer_amount` resources to the factory with no waste as factory's have infinite cargo.

If there is a robot there, by game rules and code there should only be one. We transfer `transfer_amount` resources to the robot there and anything that goes over the robot's `CARGO_SPACE` is thrown out.

This transferring setup enables simultaneous transfers. Thus, two units can both simultaneously transfer to each other without wasting resources. Moreover, you can submit actions that transfer way more resources than your robot holds, but in code we will only transfer the amount your robot actually holds.

## Environment Step - Normal Phase: Handling Resource Pickup Actions

These are handled with `self._handle_pickup_actions(actions_by_type)`

We permit actions that try to pickup way more than the factory actually holds and what the robot can contain within its `CARGO_SPACE`. However in code we will only give what the factory actually holds to the robot that picks up the resources and throw away any excess that doesn't fit in the robot's cargo space.

Unlike transfer actions, these actions are not simultaneous and are run in order of robot id. We check for each action which factory is under the robot and add resources to the robot and subtract from the factory.

## Environment Step - Normal Phase: Updating Resources Water, and Power

After all actions are handled, we need to update the map. We first subtract one lichen from every tile and then clip lichen values to the range `[0, MAX_LICHEN_PER_TILE]`. Then for any tile with no lichen we set the lichen_strain there to -1.

We then refine resources according to the rate specified by `FACTORY_PROCESSING_RATE_METAL, ORE_METAL_RATIO` and `FACTORY_PROCESSING_RATE_WATER, ICE_WATER_RATIO`. The processing rate is the maximum number of raw resources (ice/ore) that can be refined. The ratio is the ratio at which ice/ore is refined into water/metal.

Every factory then loses `FACTORY_WATER_CONSUMPTION` water. Any factory with less than 0 water now is destroyed, leaving behind `FACTORY_RUBBLE_AFTER_DESTRUCTION` on each of the factory's 3x3 tiles.

After that, every robot gains power equal to `CHARGE` and capped at `BATTERY_CAPACITY`. Every factory gains power equal to `FACTORY_CHARGE` plus plant power equal to `# connected_lichen_positions * POWER_PER_CONNECTED_LICHEN_TILE`. See the [water cost computation section](#environment-water-cost-computation) for how `connected_lichen_positions` is computed. Note that plant power does not include tiles that will gain lichen this turn due to watering.

Finally, we set rubble under all factories' 3x3 tiles to 0.

## Environment Step - Normal Phase: Generating Observations

This environment is fully observable. Both teams see the same thing and see everyone's robots, the entire map etc.

In code, we use the [State](https://github.com/Lux-AI-Challenge/Lux-Design-S2/blob/main/luxai_s2/luxai_s2/state/state.py) object to iterate over all the state data and call respective `state_dict()` functions to construct an observation.

Importantly, the observations returned by `env.step` are the full observations. **Note that in the CLI tool and competition servers, the raw observations are delta observations**. Delta observations are given for any turn when `env_steps > 0`, which contain the same thing as full observations but for board information on ice, ore, metal, water, lichen, and lichen_strains, it contains sparse matrix updates instead of the full matrix.

The full observation includes

```
{
  units: { [robot id]: UnitStateDict },
  teams: { ["player_0" or "player_1"]: TeamStateDict },
  factories: { [factory id]: FactoryStateDict },
  board: BoardStateDict
  real_env_steps: int
  global_id: int # used only for forward simulation purposes
}
```

This full observation in addition to the [EnvConfig in config.py](https://github.com/Lux-AI-Challenge/Lux-Design-S2/blob/main/luxai_s2/luxai_s2/config.py) can also be used to create a State object via `State.from_obs`.

The delta observation is the same but replaces `board` with `SparseBoardStateDict`. See [state.py](https://github.com/Lux-AI-Challenge/Lux-Design-S2/blob/main/luxai_s2/luxai_s2/state/state.py) to find typings for these.
 
## Environment Water Cost Computation

To compute water cost, we first compute `connected_lichen_tiles` and `grow_lichen_tiles` sets.

Starting from the factory's 12 adjacent tiles (excluding those that have resource tiles or rubble on them) we perform a flood fill, adding tiles if they already have lichen of the same strain as the factory on them. All of these tiles form the `connected_lichen_tiles` set.

The complicated part is now figuring out `grow_lichen_tiles`, which is a super set of `connected_lichen_tiles`.

For each tile adjacent to a connected lichen tile with `MIN_LICHEN_TO_SPREAD` or more lichen that doesn't have the factory's lichen and strain on it, we check the following

1. The tile has no rubble, is not a resource tile, and is not a factory tile
2. All adjacent tiles to the tile don't have lichen strains or if they do they are the same lichen strain as produced by this factory.
3. All adjacent tiles to the tile aren't a factory tile.

If the above is satisfied, then we add the tile to the `grow_lichen_tiles` set.

Then the watering cost is `ceil(len(grow_lichen_tiles) / 10))`.
