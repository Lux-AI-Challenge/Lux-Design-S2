from luxai2022.actions import *
from luxai2022.unit import UnitType

import random

FACTION = "AlphaStrike"

def distance(pos1, pos2):
    return sum((pos1-pos2)**2)

def nearest_factory(factories, pos):
    return min(factories.values(), key=lambda f: distance(pos, f["pos"]))

# (dx, dy) = (1, 2) is moving right 1 and up 2
# DIRECTIONS = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 0)}
DIRECTIONS = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])

def from_dirs(d):
    if all(d == [0, 0]):
        return 0
    if all(d == [0, -1]):
        return 1
    if all(d == [1, 0]):
        return 2
    if all(d == [0, 1]):
        return 3
    if all(d == [-1, 0]):
        return 4
    raise ValueError("d must be one of DIRECTIONS.")

def direction_between(pos, goal):
    return min(DIRECTIONS, key=lambda d: distance(pos + d, goal))

def spawn(observation, configuration, team_id):
    info = dict(faction=FACTION, )
    locs = observation["board"]["spawns"][team_id]
    spawn_locs = [random.choice(locs) for i in range(configuration.MAX_FACTORIES)]

    return dict(faction=FACTION, spawns=spawn_locs)

def agent(observation, configuration, team_id):
    my_team = observation["team"][team_id]
    my_units = observation["units"][team_id]
    my_factories = observation["factories"][team_id]

    factory_positions = set()
    actions = dict()
    for id, factory in my_factories.items():
        factory_positions.add(tuple(factory["pos"]))
        if (factory["power"] >= configuration.ROBOTS["LIGHT"].POWER_COST and
            factory["cargo"]["metal"] >= configuration.ROBOTS["LIGHT"].METAL_COST):
            actions[id] = FactoryBuildAction(UnitType.LIGHT).state_dict()

    for id, unit in my_units.items():
        x, y = unit["pos"]

        if unit["cargo"]["ice"] > 10 or unit["cargo"]["ore"] > 10:
            if id == "unit_11":
                print("Dropping off!")
            # Bring resources to factory. Should automatically drop off
            goal = nearest_factory(my_factories, unit["pos"])["pos"]
            move_dir = direction_between(unit["pos"], goal)
            move_dir = from_dirs(move_dir)
            actions[id] = [MoveAction(move_dir).state_dict()]

        elif ((x, y) not in factory_positions and
                observation["board"]["ice"][y, x] > 0): # Mine ice
            if id == "unit_11":
                print("Mining ice!", observation["board"]["ice"][y, x])

            actions[id] = [DigAction().state_dict()] # 0 = ice

        elif ((x, y) not in factory_positions and
                observation["board"]["ore"][y, x] > 0): # Mine ore
            if id == "unit_11":
                print("Mining ore!", observation["board"]["ore"][y, x])
            actions[id] = [DigAction().state_dict()] # 1 = ore

        else: # Move randomly
            actions[id] = [MoveAction(random.randrange(len(DIRECTIONS))).state_dict()]
            if id == "unit_11":
                print("Moving randomly!")

    return actions
