# Lux AI Challenge Season 2 Beta Specifications

For documentation on the API, see [this document](https://github.com/Lux-AI-Challenge/Lux-Design-2022/blob/main/kits/). To get started developing a bot, see our [Github](https://github.com/Lux-AI-Challenge/Lux-Design-2022). This is currently in Beta, so we are looking for feedback and bug reports, if you find any issues with the code, specifications etc. please ping us on [Discord](https://discord.gg/aWJt3UAcgn) or post a [GitHub Issue](https://github.com/Lux-AI-Challenge/Lux-Design-2022/issues)


## Background

As the sun set on the world an array of lights dotted the once dark horizon. With the help of a [brigade of toads](https://www.kaggle.com/competitions/lux-ai-2021/discussion/294993), Lux had made it past the terrors in the night to see the dawn of a new age. Seeking new challenges, plans were made to send a forward force with one mission: terraform Mars!


## Environment

In the Lux AI Challenge Season 2, two competing teams control a team of Factory and Robots that collect resources and plant lichen, with the main objective to own as much lichen as possible at the end of the turn-based game. Both teams have complete information about the entire game state and will need to make use of that information to optimize resource collection, compete for scarce resources against the opponent, and grow lichen to score points.

Each competitor must program their own agent in their language of choice. Each turn, each agent gets 3 seconds to submit their actions, excess time is not saved across turns. In each game, each player is given a pool of 60 seconds that is tapped into each time the agent goes over a turn's 3-second limit. Upon using up all 60 seconds and going over the 3-second limit, the agent freezes and loses automatically.

The rest of the document will go through the key features of this game.

## The Map

The world of Lux is represented as a 2d grid. Coordinates increase east (right) and south (down). The map is always a square and is 48 tiles long. The (0, 0) coordinate is at the top left. The map has various features including [Raw Resources](#resources) (Ice, Ore), [Refined Resources](#resources) (Water, Metal), [Robots](#robots) (Light, Heavy), [Factories](#factories), [Rubble](#movement-collisions-and-rubble), and [Lichen](#lichen). The map also includes a schedule of martian weather events impacting the environment discussed below. Code wise, the coordinate (x, y) in a map feature such as rubble is indexed by `board.rubble[x][y]` for ease of use.

In order to prevent maps from favoring one player over another, it is guaranteed that maps are always symmetric by vertical or horizontal reflection. Each player will start the game by bidding on an extra factory, then placing several Factories and specifying their starting resources. See the [Starting Phase](#starting-phase) for more details.

## Weather Events

There are 4 kinds of weather events that occur on a predetermined schedule (given to the players at the start of the game, between 3 and 5 events). These events last 1 - 30 turns with marsquakes lasting 1 - 5 turns.

* Marsquake - All robots generate rubble under them every turn (1/turn light, 10/turn heavy)
* Cold snap - 2x power consumption
* Dust storm - 0.5x power gain
* Solar flare - 2x power gain

## Day/Night Cycle

The Day/Night cycle consists of a 50 turn cycle, the first 30 turns being day turns, the last 20 being night turns. During the day, solar panels replenish the power of all robots but during the night robots power is not recharged. Factories generate power each turn regardless.

## Resources

There are two kinds of raw resources: Ice and Ore which can be refined by a factory into Water or Metal respectively. These resources are collected by Light or Heavy robots, then dropped off once a worker transfers them to a friendly factory, which then automatically converts them into refined resources at a constant rate. Refined resources are used for growing lichen (scoring points) as well as building more robots. Lastly, factories will process ice and ore 5 units at a time without wasting any. E.g. if a factory has 8 ore, it will refine 5 ore into 1 metal and leave 3 ore leftover; if a factory has 7 ice, it will refine 6 ice into 3 water and leave 1 ice leftover.

<table>
  <tr>
   <td><strong>Raw Type</strong>
   </td>
   <td><strong>Factory Processing Rate</strong>
   </td>
   <td><strong>Refined Type</strong>
   </td>
   <td><strong>Processing Ratio</strong>
   </td>
  </tr>
  <tr>
   <td>Ice
   </td>
   <td>50/turn
   </td>
   <td>Water
   </td>
   <td>2:1
   </td>
  </tr>
  <tr>
   <td>Ore
   </td>
   <td>50/turn
   </td>
   <td>Metal
   </td>
   <td>5:1
   </td>
  </tr>
</table>

## Starting Phase

During the first turn of the game, each player is given the map, starting resources (`N` factories and `N*100` water and ore), and are asked to bid on an extra factory. Each 1 bid removes 1 water and 1 ore from that player's starting resources. Each player responds in turn 1 with their bid. Whichever player places the highest bid loses X water and ore from their starting resources and receives an extra factory to place. If both players tie, neither player is awarded an extra factory. Players that do not win the bid do not lose any starting resources.

During the next N+1 turns of the game each player may select any location on their half of the map and send the FactoryBuild action to build a Factory at a specified location with the specified starting metal, water, and 100 power. After N + 1 turns any remaining factories and any resources not placed are lost. 

_Strategy Tip_: The player who lost the bid has an extra turn where they don’t have to place a factory. Skipping a turn earlier will usually provide a larger advantage.


## Actions

[Robots](#robots) and [Factories](#factories) can perform actions each turn given certain conditions and enough power to do so. In general, all actions are simultaneously applied and are validated against the state of the game at the start of a turn. Each turn players can give an action to each factory and a queue of actions to each robot. 

[Robots](#robots) always execute actions in the order of their current action queue. [Robot](#robots) actions can also be configured to be repeated, meaning once the action is executed the action is replaced to the back of the action queue instead of being removed.

Submitting a new action queue for a robot requires the robot to use additional power to replace it's action queue. It costs an additional 1 power for Lights, an additional 10 power for Heavies. The new action queue is then stored and wipes out what was stored previously. If the robot does not have enough power, the action queue is simply not replaced.

The next few sections describe the [Robots](#robots) and [Factories](#factories) in detail.

## Robots

There are two robot types, Light and Heavy. Every robot has an action queue and will attempt to execute the action at the front of the queue.

<table>
  <tr>
   <td>Dimension
   </td>
   <td>Light
   </td>
   <td>Heavy
   </td>
   <td>Factory
   </td>
  </tr>
  <tr>
   <td>Cargo Space
   </td>
   <td>100
   </td>
   <td>1000
   </td>
   <td>Infinite
   </td>
  </tr>
  <tr>
   <td>Battery Capacity
   </td>
   <td>150
   </td>
   <td>3000
   </td>
   <td>Infinite
   </td>
  </tr>
  <tr>
   <td>Power Charge (during day)
   </td>
   <td>1
   </td>
   <td>10
   </td>
   <td>50* all the time
   </td>
  </tr>
</table>

### Light and Heavy Robots

Light and Heavy Robots share the same set of actions / action space. However, in general, heavy robots generally accomplish 10x more with their actions but their actions will cost 20x more power.


### Actions

* Move - Move the robot in one of 5 directions, North, East, South, West, Center. Moving center costs no power.
* Transfer -  Send any amount of a single resource-type (including power) from a robot’s cargo to an orthogonally adjacent tile or to the tile it is standing on. If a robot is on the target tile, it will receive the transferred resources up to the robot’s cargo capacity. If the target tile is a friendly factory tile, the factory receives all the transferred resources. If the receiving entity can't receive all transferred resources due to space limitations, then the overflow resources are wasted. Factories are given preference over other robots in receiving resources from transfers.
    * Algorithmically, we perform the following procedure. The environment creates transfer requests for every robot that wants to transfer and remove the specified resources from the robot’s cargo. All transfer requests are attempted to be fulfilled, and any excess caused by not enough cargo space (either robot has no space, or too many robots transferring to the same robot and go over the max capacity), is then wasted.
* Pickup - When on top of any factory tile (there are 3x3 per factory), can pick up any amount of power or any resources. Preference is given to robots with lower robot IDs.
* Dig - Does a number of things depending on what tile the robot is on top of
    * Rubbleless resource tile - gain raw resources (ice or ore)
    * Rubble - reduce rubble by 1 if light, 10 if heavy
    * Lichen - reduce lichen value by 10 if light, 100 if heavy
* Self destruct - Destroys the robot on the spot, which creates rubble.
* Recharge X - the robot waits until it has X power. In code, the recharge X command is not removed from the action queue until the robot has X power.
* Repeat - This is not an explicit action but is a boolean/bit that can be added to each action. It tells the robot to append the action the robot just took to the end of the action queue. When set to False, executed actions are not appended back and are removed from the queue.

The following table summarizes the configurations.

<table>
  <tr>
   <td>
Action
   </td>
   <td>Light
   </td>
   <td>Heavy
   </td>
  </tr>
  <tr>
   <td>Move
   </td>
   <td>1 power
   </td>
   <td>20 power + rubble value of target square
   </td>
  </tr>
  <tr>
   <td>Transfer
   </td>
   <td>0 power
   </td>
   <td>0 power
   </td>
  </tr>
  <tr>
   <td>Pickup
   </td>
   <td>0 power
   </td>
   <td>0 power
   </td>
  </tr>
  <tr>
   <td>Dig
   </td>
   <td>5 power (1 rubble removed, 2 resources gain, 10 lichen value removed)
   </td>
   <td>100 power (10 rubble removed, 20 resource gain, 100 lichen value removed)
   </td>
  </tr>
  <tr>
   <td>Self Destruct
   </td>
   <td>10 power
   </td>
   <td>100 power
   </td>
  </tr>
  <tr>
   <td>Recharge X
   </td>
   <td>0 power
   </td>
   <td>0 power
   </td>
  </tr>
</table>

### Movement, Collisions and Rubble

Each square on the map has a rubble value which affects how difficult that square is to move onto. Rubble value is an integer ranging from 0 to 100 inclusive. The exact power required to move into a square with rubble can be found on the table above. Rubble can be removed from a square by a light or heavy robot by executing the dig action while occupying the square.

This environment also has robot collisions. Robots which move onto the same square on the same turn can be destroyed and add rubble to the square according the following rules:

* Heavy robots that end their turn on a square with only other light robots, it will destroy all the light robots and leave the single heavy robot unaffected.
* If two robots of the same weight end their turn on the same square, and one of them did not move in the previous turn, the stationary robot is destroyed. If both robots moved, both robots are destroyed.

Each light robot destroyed in this way adds 1 rubble. Each heavy robot destroyed in this way adds 10 rubble. (same values as marsquakes and self destructs).

Lastly, any addition of rubble onto a tile with [Lichen](#lichen) on it will automatically remove all of the lichen on that tile.

## Factories

A factory is a building that takes up 3x3 tiles of space. Robots created from the factory will appear at the center of the factory. Allied robots can move onto one of the factory's 9 tiles, but enemies cannot.

Each turn a factory will automatically:

* Gain 50 power (regardless of day or night)
* Convert up to 50 ice to 25 water 
* Convert up to 50 ore to 10 metal 
* Consume 1 water

If there is no water left, the nuclear reactor that powers the factory will explode, destroying the factory and leaving behind 50 rubble on each of the 3x3 tiles.

Each factory can perform one of the following actions

* Build a light robot
* Build a heavy robot
* Grow lichen - Waters lichen around the factory, costing `ceil(connected lichen tiles / 10)` water

The following is the cost to build the two classes of robots. Note that also robots when built will have their battery charged up to the power cost.

<table>
  <tr>
   <td>
Robot Type
   </td>
   <td>Metal Cost
   </td>
   <td>Power Cost
   </td>
  </tr>
  <tr>
   <td>Light Robot
   </td>
   <td>10
   </td>
   <td>50
   </td>
  </tr>
  <tr>
   <td>Heavy Robot
   </td>
   <td>100
   </td>
   <td>500
   </td>
  </tr>
</table>

## Lichen

At the end of the game, the amount of lichen on each square that a player owns is summed and whoever has a higher value wins the game. 

At the start, factories can perform the water action to start or continue lichen growing. Taking this action will seed lichen in all orthogonally adjacent squares to the factory if there is no rubble present (total of 3*4=12). Whenever a tile has a lichen value of 10 or more and is watered, it will spread lichen to adjacent tiles without rubble, resources, or factories and give them lichen values of 1. The amount of water consumed by the water action grows with the number of tiles with lichen on them connected to the factory according to `ceil(# connected lichen tiles / 10)`. In each tile a maximum of 100 lichen value can be stored.

All factories have their own special strains of lichen that can’t mix, so lichen tiles cannot spread to tiles adjacent to lichen tiles from other factories. Algorithmically, if a tile can be expanded to by two lichen strains, neither strain expands to there. This is for determinism and simplified water costs.

When rubble is added to a tile, that tile loses all lichen.

If a number of lichen tiles get disconnected from your factory (due to some rubble being added to a tile or being digged out), they cannot be watered (and thus will lose 1 lichen value) until connected again through lichen tiles.

At the end of each turn, all tiles that have not been watered lose 1 lichen.

## Game Resolution order

To help avoid confusion over smaller details of how each turn is resolved, we provide the game resolution order here and how actions are applied.

Actions in the game are first all validated against the current game state to see if they are valid. Then the actions, along with game events, are resolved in the following order and simultaneously within each step

1. Agents submit actions for robots, overwrite their action queues
2. Transfer resources and power
3. Pickup resources and power (in order of robot id)
4. Digging, self-destruct actions (removing and adding rubble)
5. Movement and recharge actions execute, then collisions are resolved
6. Factories that watered their tiles grow lichen
7. Robot Building
8. Factories refine resources
9. Power gain (if started during day for robots)

## Win Conditions

After 1000 turns, the winner is whichever team has the most lichen value on the map. If any team loses all of their factories, they automatically lose and the other team wins. If the lichen value is tied or both teams lose all their factories at the same time, then the game ends in a draw.

## Note on Game Rule Changes

Our team at the Lux AI Challenge reserves the right to make any changes on game rules during the course of the competition. We will work to keep our decision-making as transparent as possible and avoid making changes late on in the competition.
