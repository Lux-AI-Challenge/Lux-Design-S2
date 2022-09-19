# Lux AI Season 2 Specifications

Make sure you checked out our [Getting Started](https://github.com/Lux-AI-Challenge/Lux-Design-2021/#getting-started) section to setup your programming environment.

## Background

Technology has advanced to the point that Mars colonization is possible through planetary-scale terraforming. However, Earth has split into 4 factions with different perspectives on what to do with mars! Will you help expand the terraforming efforts, or will you seek out to put a stop to the terraforming?

## Environment

In the Lux AI Challenge Season 2, two competing teams control a team of [Units](#Units) and [Factories](#Factories) that collect resources to help build more units and grow lichen on the surface of Mars, with the main objective to have more lichen planted than the other team. Both teams have complete information about the entire game state and will need to make use of that information to optimize resource collection, compete for scarce resources against the opponent, and grow lichen for points.

Each competitor must program their own agent in their language of choice. Each turn, your agent gets 3 seconds to submit their actions, excess time is not saved across turns. In each episode, you are given a pool of 60 seconds that is tapped into each time you go over a turn's 3-second limit. Upon using up all 60 seconds and going over the 3-second limit, your agent freezes and automatically loses.

The rest of the document will go through the key features of this game.

## The Map and Time

The world in Lux is represented as a 2d grid. Coordinates increase east (right) and south (down). The map is always a 48 x 48 square.

![](https://raw.githubusercontent.com/Lux-AI-Challenge/Lux-Design-2021/master/assets/game_board.png)

The map has various features including [Resources](#Resources) (Ice, Water, Ore, Metal), [Units](#Units) ([Light](#Light), [Heavy](#Heavy)), and [Factories](#Factories)

In order to prevent maps from favoring one player over another, it is guaranteed that maps are always symmetric by vertical or horizontal reflection.

Every step in the environment is one step in time. There are a total of `1000 + n + 2` total timesteps per episode, where there are `n + 2` steps of the [Early Phase](#Early-Phase) and `1000` steps of the [Regular Phase](#Regular-Phase), both of which have **different action-spaces**

## Early Phase

Each player starts with a pool `n * 100` water and `n * 100` metal where `n` is an integer ranging between `2` and `6`. `n` is the starting number of factories each player gets to place down.

In the first turn of each episode, each player must select a faction to represent and make a bid for a potential extra factory. Each unit of a bid subtracts 1 water and metal from the starting pool. After bidding, the winning bidder gets an extra factory to place. If the bid is a tie, no one gets an extra factory.

In each of the next `n + 1` turns, both players can place a factory down on their half of the map and choose the amount of water and metal to put in the factory, taken out of the starting pool. Note that the winning bidder will end up placing a factory every turn whereas the loser can skip one turn to watch the opponent first before placing their factories. Any water and metal not distributed to placed factories is thrown away.

## Regular Phase

The next `1000` turns comprises the regular phase where both players control their units and factories to try and grow more lichen.

The rest of the specifications details the regular phase

## Weather Events

There are 4 kinds of weather events that occur on a predetermined schedule (given to the players at the start of the game, with between 3 and 5 events per episode). These events will each last for a number of turns.

- Marsquake - All robots generate [rubble](#Rubble) under them every turn (1/turn for light, 10/turn for heavy units). Lasts 1 to 5 turns.
- Cold snap - 2x power consumption. Lasts 10 to 30 turns.
- Dust storm - 0.5x power gain. Lasts 10 to 30 turns.
- Solar flare - 2x power gain. Lasts 10 to 30 turns.

## Resources

There are two kinds of raw resources: Ice and Ore which can be refined by a factory into Water or Metal respectively. These resources are collected by Light or Heavy units, then dropped off once a worker transfers them to a friendly factory, which then automatically converts them into refined resources at a constant rate. 

| Raw Type|  Factory Processing Rate | Refined Type | Processing Ratio |
| --- |   ---    |  --   |  -- |
| Ice |  50/turn | Water | 1:1 |
| Ore |  50/turn | Metal | 1:1 |

## Resource Collection Mechanics

As an action, a Light or Heavy robot located on a raw resource tile can send the Dig action. Whichever raw resource the robot is located on will be added to that robots inventory at their mining rateup to their cargo capacity.


## Actions

## Factories

## Units

## Rubble

## Movement and Collisions


## Power

Factories and Robots require power to take any action. Power can be transferred between robots using the `Transfer` action and can be picked up from any factory with the `Pickup` action (only while on top of a friendly factory).

## Lichen

At the end of the game, the amount of lichen on each square that a player owns is summed and whoever has a higher value wins the game. 

At the start, factories can take the water action to start or continue lichen growing. Taking this action will seed lichen in all orthogonally adjacent squares to the factory if there is no rubble present. Whenever a tile has a lichen value of 20 or more and is watered, it will spread lichen to adjacent tiles without rubble, factories, or resources and give them lichen values of 1. The amount of water consumed by the water action grows with the number of tiles with lichen on them connected to the factory according to ceil(# connected lichen tiles / 10). 

All factories have their own special strains of lichen that can’t mix, so lichen tiles cannot spread to tiles adjacent to lichen tiles from other factories (in in such a manner that tiles would be adjacent)

When rubble is added to a tile, that tile loses all lichen.

If a number of lichen tiles get disconnected from your factory (due to some rubble being added to a tile), they cannot be watered (and thus will lose 1 lichen value). If the blocking tiles are cleared of rubble the lichen field can reconnect.

At the end of each turn, all tiles that have not been watered lose 1 lichen.


## Day/Night Cycle

(it wouldn't be a lux ai challenge without some kind of day/night scenario!)

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
10. Win Conditions
11. After 1000 regular phase turns, the winner is whichever team has the most lichen value on the map. If that is a tie, then whichever team has the most power across all total robots.

## Note on Game Rule Changes
Our team at the Lux AI Challenge reserves the right to make any changes on game rules during the course of the competition. We will work to keep our decision-making as transparent as possible and avoid making changes late on in the competition.
