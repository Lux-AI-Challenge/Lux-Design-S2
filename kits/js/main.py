import json
from subprocess import Popen, PIPE
from threading  import Thread
from queue import Queue, Empty
from collections import defaultdict
from argparse import Namespace
import atexit
import os
import sys
agent_processes = defaultdict(lambda : None)
t = None
q_stderr = None
q_stdout = None
import time
def cleanup_process():
    global agent_processes
    for agent_key in agent_processes:
        proc = agent_processes[agent_key]
        if proc is not None:
            proc.kill()
def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()
def agent(observation, configuration):
    """
    a wrapper around a non-python agent
    """
    global agent_processes, t, q_stderr, q_stdout

    agent_process = agent_processes[observation.player]
    ### Do not edit ###
    if agent_process is None:
        if "__raw_path__" in configuration:
            cwd = os.path.dirname(configuration["__raw_path__"])
        else:
            cwd = os.path.dirname(__file__)
        agent_process = Popen(["node", "main.js"], stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=cwd)
        agent_processes[observation.player] = agent_process
        atexit.register(cleanup_process)

        # following 4 lines from https://stackoverflow.com/questions/375427/a-non-blocking-read-on-a-subprocess-pipe-in-python
        q_stderr = Queue()
        t = Thread(target=enqueue_output, args=(agent_process.stderr, q_stderr))
        t.daemon = True # thread dies with the program
        t.start()
    data = json.dumps(dict(obs=json.loads(observation.obs), step=observation.step, remainingOverageTime=observation.remainingOverageTime, player=observation.player, info=configuration))
    agent_process.stdin.write(f"{data}\n".encode())
    agent_process.stdin.flush()

    agent1res = (agent_process.stdout.readline()).decode()
    while True:
        try:  line = q_stderr.get_nowait()
        except Empty:
            # no standard error received, break
            break
        else:
            # standard error output received, print it out
            print(line.decode(), file=sys.stderr, end='')
    if agent1res == "":
        return {}
    return json.loads(agent1res)

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
        actions = agent(observation, dict(env_cfg=configurations))
        # send actions to engine
        print(json.dumps(actions))