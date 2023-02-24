# ChangeLog

### v2.1.9

Fix bug where setuptools was causing runtime errors
### v2.1.8

fix bug on mac where non python kits assumed a windows platform

store replay seed and player paths on CLI tool.

visualizer shows the player paths as well as seed on kaggle and local replays


### v2.1.7
removed omegaconf as dependency


### v2.1.6

some bug fixes

### v2.1.5

verbosity is nicer and more bug fixes

### v2.1.4

Fix bug with action formatter not handling lists

### v2.1.3

Fix bug where calling reset with no seed crashes the engine.

### v2.1.2

Cast all numbers to ints, ensure all observations only contain ints, no floats.

### v2.1.1

Remove max episode timesteps from gym registration of the Lux AI env. Expect user to specify themselves


### v2.1.0

Added [advanced_specs](https://github.com/Lux-AI-Challenge/Lux-Design-S2/blob/main/docs/advanced_specs.md) document that goes over CPU engine code in depth

Fix bug where repeated actions had their `n` value reset to 1. 

Actions specify `n` and `repeat` but they are now slightly modified. `n` is the current execution count of an action, meaning the robot will try to execute it `n` times (with unsuccesful attempts due to e.g. lack of power not counting). `n` is decremented for each successful execution. `repeat` is no longer a boolean and is now an int. if `repeat == 0`, then after `n` goes to 0, the action is removed from the action queue, If `repeat > 0`, then after `n` goes to 0, we **recycle** the action to the back of the action queue and **set n = repeat**. 

Fix bug where two heavies entering a tile can both get destroyed if a single light unit there has more power.

Fix bug where clearing action queues with an empty action queue `[]` was not permitted.

Java kit and updated JS, C++ kits are merged in.
### v2.0.6

Fix bug where no seed provided to CLI meant no random maps

Fixed bug where plant power uses the growing lichen positions, not the connected ones.

Fix local config. Self destruct cost of lights is 10, not 5 (matching the specs document). Fixed sample env cfg json as well

### v2.0.2

Fix bug with seeding the map generation where the sampled seed is above allowed range

### v2.0.0

Official release!

Major Engine Changes
- Weather is removed
- Each tile of lichen owned and connected to a factory gives 1 power to the factory each turn
- Actions in an action queue can specify both the number of times to repeat it directly, as well as whether to put it back to the end of the action queue once exhausted
- When digging out lichen, if the final lichen is dug out then rubble equal to `DIG_RUBBLE_REMOVED` is added to the tile.
- When handling collisions, if two units of the same weight class move onto the same tile, previously they both were destroyed. Now, all units with less power are destroyed and the unit with the most power in the collision loses half of the power of the unit with the 2nd most power.
- Map generation has been updated to include resource distribution types, a matrix of low/high ore and low/high ice. Low = ~15 tiles, High = ~35 tiles.

Configuration Changes
- Light units rubble cost is floor(0.05 * rubble) now.
- Heavy units dig cost is 60 power instead of 100.
- It takes 20 lichen to grow to new tiles instead of 10 now.

New Features
- You can reconstruct a LuxAI_S2 state object from a complete observation or from observations with sparse encoded board updates and forward simulate

New minimal visualizer (Lux-Eye-S2)

Repo Changes
- Repository has been reorganized to accomodate more packages, specifically the gpu version

Observation Space Changes
- Add `global_id` so state can be reconstructed perfectly
- Teams: add `bid` datapoint


### v1.1.6
- Fix bugs related to tournament tool and not printing some numbers as well as concurrent configuration not used
### v1.1.5

- Fix bugs related to the tournament CLI tool where it required to save replays and didn't run Win/Loss/Tie ranking systems
### v1.1.4

- Fix bug where lichen could grow to tiles adjacent to factories
- Fix bug where if one factory fails to build a unit, all other factories fail to build units
- Fix bugs for lichen growth on border overlapping a little
- Fix bug with local CLI using different time setup to kaggle.
- Transfers are irrelevant of unit ID now, and is completely simultaneous.
- Fix bug in factory placement where if placement failed due to using too much metal, we set metal to `init_water`
- Fix bug where printed collided agents is incorrect and shows previous collided units in the same turn
- Fix bug for windows on python 3.7 with asyncio
- Transfers and pickups happen at the end of a turn just before refining and powering
- Fix engine crash where erroring on turn 0 crashes engine
- Lichen growing positons are now computed after everything happens (dig, self destruct, movement)
### v1.1.3

- Fix bug with lichen frontiers being computed incorrectly
- Fix bug where units could transfer more than they held into factories
- Fix bug when traces were printed it errored instead.
- Added some aggregate meta data for preparation of factions
- Fixed critical bug in starter kit where units wasted their time expending energy sending a move center action
### v1.1.2
- Fix bug with C++ agents not running with CLI tool

### v1.1.1

- Allow "negative" bids to bid to go second instead of first in factory placement. The winning bidder is whoever has the higher absolute value bid. The winning bidder now loses water and metal equal to the absolute value of their bid.
- Fixed bug in python kit that adds 1 to the water cost calculation in the `lux/factory.py` file
- Removed an extra `1` in the state representation of move actions left from old code that allowed robots to move more than 1 tile.
- Fixed bug where move center actions weren't repeated.
- Fix visualizer bug where bid for player_1 showed up as player_0's bid
- Fixed repeats. Now repeat `n` times means the action is repeatedly executed `n` times before the next action in the queue. Repeat `-1` means to move the action to the end of the queue.

### v1.1.0

**Most Important Changes:**
- Switch to x,y indexing for all map and position related data. E.g. any code that did `board[y][x]` should be `board[x][y]` now. We will likely not change this ever again and will keep this for the rest of beta and the official competition. For python competitors a simple switch is to just do `board.T[y][x]` which is equivalent to `board[x][y]`.
- Maps are asymmetric now, teams now bid for being first to place a factory instead of a new factory
- Please download the new python starter kits! There are changes to the observations so old code will not work. Moreover, the action space is now different. Instead of specifying whether an action in an action queue should be repeated, you now specify -1 for infinite repeating, and n for n repeats.

Rest of changelog for v1.1.0:

Environment:
- Assymmetric Maps are used now
- Bids are now for who goes first, not who gets an extra factory. If bid is tied, player 0 goes first. This early phase is kept parallel for simplicity. Instead, when it is not your turn to place a factory you must skip your turn. Subsequent turns after the starting phase are still in parallel.
- renamed `board.spawns` -> `board.valid_spawns_mask`. Stores the same information (locations on map that you can spawn a factory on). This changes over time based on where new factories are placed.
- moved `env_cfg.UNIT_ACTION_QUEUE_POWER_COST` to their respective spots under `env_cfg.ROBOTS` e.g. `env_cfg.ROBOTS["LIGHT"].ACTION_QUEUE_POWER_COST`
- Factories can spawn anywhere, but must not overlap any resource or factory.
- Switch to x,y indexing for all map and position related data
- Bumped up initial water/metal resources per factory to 150 each (old: 100)
- Bumped up initial power per factory to 1000 (old: 100)
- Lichen requires 10 units before being able to expand now (old: 20)
- Lichen grows from any square cardinally adjacent to a factory tile now (so an factory without any surrounding rubble will immediately grow 4*3=12 lichen tiles)
- Lichen no longer grows over ice/ore tiles
- Units can transfer any amount and wont have an action cancelled. Environment will internally clip it so that unit only transfers as much as they can and target unit only receives as much as it can. Excess given to a unit is wasted. 
- Factories spawned can be spawned with any amount of resources and the placement won't be cancelled. Instead a warning is given and only the maximum of resources left will be used.
- Factories must spawn 6 or more tiles away from another
- Heavies/Lights dig 2x more rubble per action (Heavies: 10->20, Lights: 1->2)

Visualizer:
- x,y indexing fixes
- fix bug where slider over weather goes off the weather bar

Python Kits:
- Updated to accept some new observation entires (`teams[player].place_first`, `board.valid_spawns_mask`), and removed old ones (`board.spawns`)
- Added utility function to determine if it is your team's turn to place a factory or not

Bug Fixes:
- Lichen water cost costed 1 extra
- Fix bug where added rubble doesn't remove all lichen underneath
- Fix bug where move center costs power
- Fix extra line in stderr logging
- Potential fix of Windows issues with verbose error logging
- Fix bug in lichen growth where lichen would grow to new tiles if there is just 1 lichen, it should be 20
- Fix bug where agent could create infinite robots without spending metal via repeated pickups and builds
- Remove irrelevant rubble information stored in marsquake config.
- Fix bug where rubble goes onto factories when there is a unit on top during marsquakes
- Fix bugs in observation space of off by one error in map shapes
- Clarify in specs and fixed bug where added rubble didn't remove lichen under it.

CLI Tool:
- Fix bug where Java agents can't be run

### v1.0.6
- Fix bug where game ends at turns < 1000 (kaggle-environments bug)
- Fixed bug with self-destruct actions not being validated or added
- Log unit ids that collided. E.g. `14: 1 Units: (unit_11) collided at 33,44 with [1] unit_9 UnitType.HEAVY at (33, 44) surviving`
- Fixed bug where power costs were recomputed after state change in a single step, causing potentially negative energy in edge cases

### v1.0.5

Environment:
- Fixed bug where it outputs TypeError and can't serialize np.int32. This is caused by windows systems using np.int32 (but utils didn't convert it to a int)
- Fixed bug where factories could overlap
- More informative error message when user submits action instead of action queue
- Fixed bug with weather events potentially overlapping
- Fixed bug where units were not capped by their battery capacity
- Fixed bug where we do not invalidate off map transfers

Visualizer:
- Fixed small offset bug in weather schedule display
### v1.0.4

Initial beta release. Good luck!