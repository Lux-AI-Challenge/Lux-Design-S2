from luxai2022.actions import *
from luxai2022.unit import UnitType

import random

FACTION = "AlphaStrike"

def distance(pos1, pos2):
    return (pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2

def nearest_factory(factories, pos):
    return min(factories.values(), key=lambda f: distance(pos, f["pos"]))

# (dx, dy) = (1, 2) is moving right 1 and up 2
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
    my_team = observation["teams"][team_id]
    my_units = observation["units"][team_id]
    my_factories = observation["factories"][team_id]

    factory_positions = set()
    actions = dict()
    for id, factory in my_factories.items():
        if id == "factory_0":
            print(id, factory["cargo"])
        factory_positions.add(tuple(factory["pos"]))
        if (factory["power"] >= configuration.ROBOTS["LIGHT"].POWER_COST and
          factory["cargo"]["metal"] >= configuration.ROBOTS["LIGHT"].METAL_COST):
            actions[id] = FactoryBuildAction(UnitType.LIGHT).state_dict()

    for id, unit in my_units.items():
        x, y = unit["pos"]
        cargo_space = configuration.ROBOTS[unit["unit_type"]].CARGO_SPACE

        if unit["cargo"]["ice"] + unit["cargo"]["ore"] >= cargo_space:
            if id == "unit_11":
                print("Dropping off!", x, y, unit["cargo"])
            # Bring resources to factory.
            goal = nearest_factory(my_factories, unit["pos"])["pos"]
            dir = direction_between(unit["pos"], goal)
            dir = from_dirs(dir)
            if distance(unit["pos"], goal) <= 8:
                if unit["cargo"]["ice"] > unit["cargo"]["ore"]:
                    actions[id] = [TransferAction(dir, 0, unit["cargo"]["ice"]).state_dict()]
                else:
                    actions[id] = [TransferAction(dir, 1, unit["cargo"]["ore"]).state_dict()]
            else:
                actions[id] = [MoveAction(dir).state_dict()]

        elif (observation["board"]["ice"][y][x] > 0): # Mine ice
            actions[id] = [DigAction().state_dict()] # 0 = ice

        elif (observation["board"]["ore"][y][x] > 0): # Mine ore
            actions[id] = [DigAction().state_dict()] # 1 = ore

        else: # Move randomly
            actions[id] = [MoveAction(random.randrange(len(DIRECTIONS))).state_dict()]
            if id == "unit_11":
                print("Moving randomly!")

    return actions
