from luxai2022.actions import *
from luxai2022.unit import UnitType

import random

MY_TEAM_KEY = None

def distance(pos1, pos2):
    return (pos1.x - pos2.x) ** 2 + (pos1.y - pos2.x)**2

def nearest_factory(factories, pos):
    return min(factories, key=lambda f: distance(pos, f.pos))

# (dx, dy) = (1, 2) is moving right 1 and up 2
DIRECTIONS = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 0)}
def direction_between(pos, goal):
    return min(DIRECTIONS, key=lambda d: distance(pos + DIRECTIONS[d], goal))

def agent(observation, configuration):
    # How do we find which team is ours?
    # TODO: Include this in observation

    my_team = teams["factories"][MY_TEAM_KEY]
    my_units = observation["units"][MY_TEAM_KEY]
    my_factories = observation["factories"][MY_TEAM_KEY]

    actions = dict()
    for factory in my_factories:
        if (factory.cargo.metal >= configuration.ROBOTS["LIGHT"].METAL_COST and
            factory.cargo.power >= configuration.ROBOTS["LIGHT"].POWER_COST):
            actions[factory.unit_id] = FactoryBuildAction(UnitType.LIGHT)

    for unit in my_units:
        if unit.cargo.ice > 10 or unit.cargo.ore > 10:
            # Bring resources to factory. Should automatically drop off
            goal = nearest_factory(my_factories, unit.pos)
            move_dir = direction_between(unit.pos, goal)
            actions[unit.id] = MoveAction(move_dir)

        elif observation["resources"]["ice"][unit.pos] > 0: # Mine ice
            if unit.unit_type == UnitType.LIGHT:
                pickup = configuration.ROBOTS["LIGHT"].DIG_RESOURCE_GAIN
            else:
                pickup = configuration.ROBOTS["HEAVY"].DIG_RESOURCE_GAIN

            actions[unit.id] = PickupAction(0, pickup, repeat=True) # 0 = ice

        elif observation["resources"]["ore"][unit.pos] > 0: # Mine ore
            if unit.unit_type == UnitType.LIGHT:
                pickup = configuration.ROBOTS["LIGHT"].DIG_RESOURCE_GAIN
            else:
                pickup = configuration.ROBOTS["HEAVY"].DIG_RESOURCE_GAIN

            actions[unit.id] = PickupAction(1, pickup, repeat=True) # 1 = ore

        else: # Move randomly
            actions[unit.id] = MoveAction(random.choice(DIRECTIONS))

        # How to append the actions to the queue?
        # TODO: Figure that out.
        return actions
