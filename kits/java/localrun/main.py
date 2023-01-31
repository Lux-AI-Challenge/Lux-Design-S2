import json
from subprocess import Popen, PIPE
from threading  import Thread
from queue import Queue, Empty
from collections import defaultdict
from argparse import Namespace
import atexit
import io, os, time, sys
import signal
import platform

agent_processes = defaultdict(lambda : None)
t = None
q_stderr = None
q_stdout = None
def cleanup_process():
    global agent_processes
    for agent_key in agent_processes:
        proc = agent_processes[agent_key]
        if proc is not None:
            proc.terminate()
def dump_log(text: str, filepath: str = 'DUMP.txt') -> None:
    f = open(filepath, "a")
    f.write(text)
    f.close()
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

        cwd = os.getcwd()

        if platform.system() == "Windows":
            agent_process = Popen(['java', '-jar', 'JavaBot.jar'], stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=cwd, shell=True)
        else:
            agent_process = Popen(['java -jar JavaBot.jar'], stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=cwd, shell=True)

        agent_processes[observation.player] = agent_process
        atexit.register(cleanup_process)
        signal.signal(signal.SIGTERM, cleanup_process)
        signal.signal(signal.SIGINT, cleanup_process)

        # following 4 lines from https://stackoverflow.com/questions/375427/a-non-blocking-read-on-a-subprocess-pipe-in-python
        q_stderr = Queue()
        t = Thread(target=enqueue_output, args=(agent_process.stderr, q_stderr))
        t.daemon = True # thread dies with the program
        t.start()

    obs = json.loads(observation.obs)

    # info = json.loads(observation.info)
    jsonDict = dict(
        obs=obs,
        step=observation.step,
        remainingOverageTime=observation.remainingOverageTime,
        player=observation.player)
    if "env_cfg" in observation.info:
        infoDict = dict (env_cfg=observation.info["env_cfg"])
        jsonDict["info"] = infoDict

    data = json.dumps(jsonDict)
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
            # print(line.decode(), file=sys.stderr, end='')
            sys.stderr.write(line.decode())
            sys.stderr.flush()
    if agent1res == "":
        return {}

    return json.loads(agent1res)

if __name__ == "__main__":
    step = 0
    player_id = 0
    configurations = None
    i = 0

    totTime = 0;
    while True:
        inputs = sys.stdin.readline()
        obs = json.loads(inputs)

        observation = Namespace(**dict(step=obs["step"], obs=json.dumps(obs["obs"]), remainingOverageTime=obs["remainingOverageTime"], player=obs["player"], info=obs["info"]))

        if i == 0:
            configurations = obs["info"]["env_cfg"]

        i += 1
        actions = agent(observation, dict(env_cfg=configurations))
        # send actions to engine
        sys.stdout.write(json.dumps(actions)+'\n')
        sys.stdout.flush()


