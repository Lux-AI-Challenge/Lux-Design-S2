## Background

As the sun set on the world an array of lights dotted the once dark horizon. With the help of a brigade of toads, Lux had made it past the terrors in the night to see the dawn of a new age. Seeking new challenges, plans were made to send a forward force with one mission: colonize Mars!


## Environment

In the Lux AI Challenge Season 2, two competing teams control a team of Factory and Robots that collect resources and plant lichen, with the main objective to own as much lichen as possible at the end of the turn-based game. Both teams have complete information about the entire board state (but not any queued actions) and will need to make use of that information to optimize resource collection, compete for scarce resources against the opponent, and build lichen to score points.

Each competitor must program their own agent in their language of choice. Each turn, each agent gets 3 seconds to submit their actions, excess time is not saved across turns. In each game, each player is given a pool of 60 seconds that is tapped into each time the agent goes over a turn's 3-second limit. Upon using up all 60 seconds and going over the 3-second limit, the agent freezes and loses automatically.

The rest of the document will go through the key features of this game.


## The Map

The world of Lux is represented as a 2d grid. Coordinates increase east (right) and south (down). The map is always a square and can be 64 tiles long. The (0, 0) coordinate is at the top left.

The map has various features including Raw Resources (Martian Ice, Metal Ore), Refined Resources (Water, Metal), Robots (Light, Heavy), Factories, Rubble, and Lichen. The map also includes a schedule of martian weather events discussed below.

In order to prevent maps from favoring one player over another, it is guaranteed that maps are always symmetric by vertical or horizontal or diagonal (TBD) reflection.

Each player will start the game by bidding on an extra factory, then placing several Factories and specifying their starting resources. See the Starting Phase for more detail.


### Weather Events

There are 4 kinds of weather events that occur on a predetermined schedule (given to the players at the start of the game, between 3 and 5 events). These events will each last for 20 turns.



* Marsquake - All robots generate rubble under them every turn (1/turn light, 10/turn heavy)
* Cold snap - 2x power consumption
* Dust storm - 0.5x power gain
* Solar flare - 2x power gain


## Resources

There are two kinds of raw resources: Martian Ice and Metal Ore which can be refined by a factory into Water or Metal respectively. These resources are collected by Light or Heavy robots, then dropped off once a worker transfers them to a friendly factory, which then automatically converts them into refined resources at a constant rate. 


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
   <td>Martian Ice
   </td>
   <td>50/turn
   </td>
   <td>Water
   </td>
   <td>10:1
   </td>
  </tr>
  <tr>
   <td>Metal Ore
   </td>
   <td>50/turn
   </td>
   <td>Metal
   </td>
   <td>10:1
   </td>
  </tr>
</table>



## Collection Mechanics

As an action, a Light or Heavy robot located on a raw resource tile can send the Dig action. Whichever raw resource the robot is located on will be added to that robots inventory at their mining rateup to their cargo capacity.


## Actions

Robots and Factories can perform actions each turn given certain conditions and enough power to do so. In general, all actions are simultaneously applied and are validated against the state of the game at the start of a turn. Each turn players can give an action to each factory, as well as up to 10 actions to robots. These new actions need to be decoded by the robot, costing power and will be stored in a robot’s action queue, wiping out what was stored previously. Each robot will wait until it has the necessary power to execute the next action in the queue. When the action queue is exhausted the robot will take no actions. Optionally, when setting the action queue the player can specify that the robot should repeat the actions. A robot repeating actions will add them to the end of the action queue after completing them.

The next few sections describe the Robots and Factories in detail.


## Starting Phase

During the first turn of the game, each player is given the map, starting resources (`N` factories and `N*100` water and ore), and are asked to bid on an extra factory. Each 1 bid removes 1 water and 1 ore from that player's starting resources. Each player responds in turn 1 with their bid. Whichever player places the highest bid loses X water and ore from their starting resources and receives an extra factory to place. If both players tie, neither player is awarded an extra factory. Players that do not win the bid do not lose any starting resources.

During the next N+1 turns of the game each player may select any location on their half of the map and send the FactoryBuild action to build a Factory at a specified location with the specified starting metal, water, and 100 power. After N + 1 turns any remaining factories and any resources not placed are lost. 

_Strategy Tip_: The player who lost the bid has an extra turn where they don’t have to place a factory. Skipping a turn earlier will usually provide a larger advantage.


## Factories

A factory is a building that takes up 3x3 tiles of space. Robots created from the factory will appear at the center of the factory. Allied robots can move onto a factory, but enemies cannot.

Each factory requires 1~2 water a turn to cool down the nuclear reactor that powers the factories. Should the factory end a turn with no water, the factory will then overheat and meltdown spewing rubble in a 7x7 square with 50 - manhattan distance. Max rubble will be added to all squares and all robots inside the factory will be destroyed.

Each turn a factory will automatically:



* Gain 25 power and consume 1 water (day and night)
* Convert up to 50 martian ice to 5 water 
* Convert up to 50 metal ore to 5 metal 

Actions:



* Build Light Robot - Builds a light robot
* Build Heavy Robot - Builds a heavy robot
* Water lichen - Costs ceil(connected lichen tiles / 10) water

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



## Robots

There are two robot types, Light and Heavy. Every robot can perform a single action each round given they have enough power to complete this action. Robots enter the world with full power.


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
   <td>50
   </td>
   <td>1500
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

Light and Heavy Robots share a set of actions. Heavy robots generally accomplish 10x more with their actions but their actions are less power efficient.


### Actions



* Move - Move the robot in one of 5 directions, North, East, South, West, Center.
    * Center: idle?
* Transfer -  Send any amount of a single resource-type (including power) from a robot’s cargo to an orthogonally adjacent tile. If a robot is on the tile, receive the transferred resources up to the robot’s cargo capacity. Excess is returned to the original robot. If the receiving robot is on top of a factory, the factory receives all the transferred resources.
    * Create transfer requests for every robot and remove from robot’s cargo
    * All transfer requests are attempted to be fulfilled, and any excess caused by not enough cargo space (either robot has no space, or too many robots transferring as well and go over max), is then wasted to the wind
        * Chaining is not allowed as a result
    * When transferring power, the power cost to transfer is lost.
* Pickup - When on top of a factory, can pick up any amount of power or any resources
    * Pickup commands completed in order of robot id. (build order basically)
* Dig - Does a number of things depending on what tile the robot is on top of
    * Rubbleless resource tile - gain raw resources
    * Rubble - reduce rubble by 1 if light, 10 if heavy
    * Lichen - reduce lichen value by 10 if light, 100 if heavy
* Recharge X - the robot waits until it has X power
* Repeat - tells the robot to instead of removing completed actions from their action queue, append the action the robot just took to the end of the action queue

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
   <td>1 power + rubble value of target square
   </td>
   <td>20 power + 5 * rubble value of target square
   </td>
  </tr>
  <tr>
   <td>Transfer
   </td>
   <td>1 power
   </td>
   <td>20 power
   </td>
  </tr>
  <tr>
   <td>Pickup
   </td>
   <td>1 power
   </td>
   <td>20 power
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
  <tr>
   <td>Repeat
   </td>
   <td>* boolean, only applicable when setting action queue, default False
   </td>
   <td>* boolean, only applicable when setting action queue, default False
   </td>
  </tr>
</table>



### Movement, Collisions and Rubble

Each square on the map has a dynamic rubble value which affects how difficult that square is to move onto. Rubble value is a non-negative integer value with no maximum but in practice generated maps will start with at most 99 rubble. The exact power required to move into a square with rubble can be found on the table above. Rubble can be removed from a square by a light or heavy robot by executing the dig action while occupying the square.

Robots which move into the same square can be destroyed and add rubble to the square according the following: 



* Heavy robots that end their turn on a square with only other light robots will destroy all the light robots and leave the heavy robot unaffected.
* If two robots of the same weight end their turn on the same square, and one of them did not move in the previous turn, the stationary robot is destroyed. If both robots move, both robots are destroyed.
* Each light robot destroyed in this way adds 1 rubble
* Each heavy robot destroyed in this way adds 10 rubble


## Power

Factories and Robots require power to take any action. Power can be transferred between robots using the `Transfer` action and can be picked up from any factory with the Pick up action (only while on top of a friendly factory).


## Lichen

At the end of the game, the amount of lichen on each square that a player owns is summed and whoever has a higher value wins the game. 

At the start, factories can take the water action to start or continue lichen growing. Taking this action will seed lichen in all orthogonally adjacent squares to the factory if there is no rubble present. Whenever a tile has a lichen value of 20 or more and is watered, it will spread lichen to adjacent tiles without rubble, factories, or resources and give them lichen values of 1. The amount of water consumed by the water action grows with the number of tiles with lichen on them connected to the factory according to ceil(# connected lichen tiles / 10). 

All factories have their own special strains of lichen that can’t mix, so lichen tiles cannot spread to tiles adjacent to lichen tiles from other factories (in in such a manner that tiles would be adjacent)

When rubble is added to a tile, that tile loses all lichen.

If a number of lichen tiles get disconnected from your factory (due to some rubble being added to a tile), they cannot be watered (and thus will lose 1 lichen value). If the blocking tiles are cleared of rubble the lichen field can reconnect.

At the end of each turn, all tiles that have not been watered lose 1 lichen.


## Day/Night Cycle

The Day/Night cycle consists of a 50 turn cycle, the first 30 turns being day turns, the last 20 being night turns. During the day, solar panels replenish the power of all robots but during the night robots power is not recharged. Factories generate power each turn regardless.


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
8. Factories refine resources and consume water
9. Power gain (if started during day for robots)


## Win Conditions

After 1000 turns (2s per turn) the winner is whichever team has the most lichen value on the map. If that is a tie, then whichever team has the most power across all total robots.

A game may end early if a team has no factories with enough metal to create a light robot.


## Note on Game Rule Changes

Our team at the Lux AI Challenge reserves the right to make any changes on game rules during the course of the competition. We will work to keep our decision-making as transparent as possible and avoid making changes late on in the competition.
