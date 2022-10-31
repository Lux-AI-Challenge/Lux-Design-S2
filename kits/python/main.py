import json
from typing import Dict
import sys
from argparse import Namespace

from agent import Agent
from lux.config import EnvConfig
from lux.kit import GameState, process_obs, to_json, from_json, process_action, obs_to_game_state
### DO NOT REMOVE THE FOLLOWING CODE ###
agent_dict = dict() # store potentially multiple dictionaries as kaggle imports code directly
def agent_fn(observation, configurations):
    """
    agent definition for kaggle submission.
    """
    global agent_dict
    step = observation.step
    
    
    player = observation.player
    remainingOverageTime = observation.remainingOverageTime
    if step == 0:
        agent_dict[player] = Agent(player)
        agent = agent_dict[player]
        agent.env_cfg = EnvConfig.from_dict(configurations["env_cfg"])
    agent = agent_dict[player]
    obs = process_obs(player, agent.game_state, step, json.loads(observation.obs))
    agent.step = step
    if step <= obs["board"]["factories_per_team"] + 1:
        actions = agent.early_setup(step, obs, remainingOverageTime)
    else:
        real_env_steps = obs["real_env_steps"]
        actions = agent.act(real_env_steps, obs, remainingOverageTime)

    return process_action(actions)

if __name__ == "__main__":
    
    def read_input():
        """
        Reads input from stdin
        """
        try:
            return input()
        except EOFError as eof:
            raise SystemExit(eof)
    step = 0
    player_id = 0
    configurations = None
    i = 0
    while True:
        inputs = read_input()
        obs = json.loads(inputs)
        
        observation = Namespace(**dict(step=obs["step"], obs=json.dumps(obs["obs"]), remainingOverageTime=obs["remainingOverageTime"], player=obs["player"], info=obs["info"]))
        if i == 0:
            configurations = obs["info"]["env_cfg"]
        i += 1
        actions = agent_fn(observation, dict(env_cfg=configurations))
        # send actions to engine
        print(json.dumps(actions))