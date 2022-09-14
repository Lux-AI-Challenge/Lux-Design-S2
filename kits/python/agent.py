import json
import sys
from typing import Dict
from lux.kit import process_obs, to_json, from_json, process_action
import numpy as np
class Agent():
    def __init__(self, player: str) -> None:
        self.player = player
        self.opp_player = ""
        self.game_state = None
        if self.player == "player_0":
            self.opp_player = "player_1"
        else:
            self.opp_player = "player_0"
        np.random.seed(0)

    def early_setup(self, step, obs, remainingOverageTime: int):
        """
        Logic here to make actions in the early game. Select faction, bid for an extra factory, and place factories
        """
        # various maps to help aid in decision making over factory placement
        rubble = obs["board"]["rubble"]
        # if ice[y][x] > 0, then there is an ice tile at (x, y)
        ice = obs["board"]["ice"]
        # if ore[y][x] > 0, then there is an ore tile at (x, y)
        ore = obs["board"]["ore"]

        if step == 0:
            # decide on a faction, and make a bid for the extra factory. 
            # Each unit of bid removes one unit of water and metal from your initial pool
            faction = "MotherMars"
            if self.player == "player_1":
                faction = "AlphaStrike"
            return dict(faction=faction, bid=10)
        else:
            # decide on where to spawn the next factory. Returning an empty dict() will skip your factory placement

            # how much water and metal you have in your starting pool to give to new factories
            water_left = obs["team"][self.player]["water"]
            metal_left = obs["team"][self.player]["metal"]
            # how many factories you have left to place
            factories_to_place = obs["team"][self.player]["factories_to_place"]
            # obs["team"][self.opp_player] has the same information but for the other team
            # potential spawnable locations in your half of the map
            potential_spawns = obs["board"]["spawns"][self.player]

            # as a naive approach we randomly select a spawn location and spawn a factory there
            spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
            return dict(spawn=spawn_loc, metal=62, water=62)

    def act(self, step: int, obs: Dict, remainingOverageTime: int):
        """
        Logic here to make actions for the rest of the game.

        Parameters
        ----------
        step - the current environment step. This is the "real" time step, starts from 0
            after all the bidding and factory placement is done.

        obs - the env observation. This is the raw dictionary based observation and it's usage is detailed below.
            The more raw format is suitable for those who may need it for ML based solutions. For programmed solutions
            an API is provided that interfaces with the observation more nicely, in addition to providing
            functions to generate action vectors

        remainingOverageTime - amount of time in seconds left to make an action in addition to the 
            2s timer per action. Time spent > 2s eats into the remainingOverageTime pool for next turn


        NOTE that step is set back to 0 in this part of the code. This act function is executed max_episode_length times.
        """
        actions = dict()
        with open("log", "w") as f:
            f.writelines(json.dumps(to_json(obs)))
        # the weather schedule, a sequence of values representing what the weather is at each real time step
        # 0 = Nothing, 1 = MARS_QUAKE, 2 = COLD_SNAP, 3 = DUST_STORM, 4 = SOLAR_FLARE
        weather_schedule = obs["weather"]
        current_weather = obs["weather"][step]

        # various maps to help aid in decision making
        rubble = obs["board"]["rubble"]

        # if ice[y][x] > 0, then there is an ice tile at (x, y)
        ice = obs["board"]["ice"]
        # if ore[y][x] > 0, then there is an ore tile at (x, y)
        ore = obs["board"]["ore"]

        # lichen[y][x] = amount of lichen at tile (x, y)
        lichen = obs["board"]["lichen"]
        # lichenStrains[y][x] = the strain id of the lichen at tile (x, y). Each strain id is
        # associated with a single factory and cannot mix with other strains. 
        # factory.strain_id defines the factory's strain id
        lichenStrains = obs["board"]["lichen_strains"]

        # units and factories for your team and the opposition team
        units = obs["units"][agent.player]
        opp_units = obs["units"][agent.opp_player]
        factories = obs["factories"][agent.player]
        opp_factories = obs["factories"][agent.opp_player]
        
        # iterate over all active factories
        for unit_id, factory in factories.items():
            if step % 4 == 0 and step > 1:
                actions[unit_id] = 0 #np.random.randint(0,2)
            else:
                actions[unit_id] = 2
        for unit_id, unit in units.items():
            pos = unit['pos']
            actions[unit_id] = np.array([0, np.random.randint(0, 5), 0, 0, 0])
        return actions

agent = None
def agent_fn(observation, configurations):
    """
    agent definition for kaggle submission.
    """
    global agent
    step = observation.step
    
    player = observation.player
    remainingOverageTime = observation.remainingOverageTime
    if step == 0:
        agent = Agent(player)
    new_game_state = process_obs(player, agent.game_state, step, json.loads(observation.obs))
    agent.game_state = new_game_state

    if step <= agent.game_state["board"]["factories_per_team"] + 1:
        actions = agent.early_setup(step, new_game_state, remainingOverageTime)
    else:
        real_env_steps = new_game_state["real_env_steps"]
        actions = agent.act(real_env_steps, new_game_state, remainingOverageTime)

    return process_action(actions)
