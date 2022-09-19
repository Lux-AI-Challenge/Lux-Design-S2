import asyncio
import json
import os.path as osp
import os
from subprocess import Popen
import time
from luxai_runner.ext_to_command import ext_to_command
from luxai_runner.logger import Logger
from luxai_runner.process import BotProcess


class Bot:
    def __init__(self, main_file_path: str, agent: str, agent_idx: str, verbose: int = 1) -> None:
        self.main_file_path = main_file_path
        self.file_ext = osp.splitext(self.main_file_path)[-1]
        if self.file_ext not in ext_to_command:
            raise ValueError(
                f"{self.file_ext} is not a known file extension, we don't know what command to use. \
                Edit https://github.com/Lux-AI-Challenge/Lux-Design-2021/blob/master/luxai_runner/ext_to_command.py \
                    to support a new extension"
            )
        self.command = ext_to_command[self.file_ext]
        self.is_python = self.file_ext == ".py"
        self.agent = agent
        self.agent_idx = agent_idx
        self.verbose = verbose
        self.proc = BotProcess(self.command, self.main_file_path, verbose=verbose)
        # timing
        self.remainingOverageTime = 10
        self.time_per_step = 2

        self.log = Logger(identifier=f"{self.agent}, {self.main_file_path}",verbosity=verbose)

    async def step(self, obs, step: int, reward: float = 0, info=dict()):
        stime = time.time()
        data = json.dumps(dict(obs=obs, step=step, remainingOverageTime=self.remainingOverageTime, player=self.agent, reward=float(reward), info=info))
        try:
            action, stderr = await asyncio.wait_for(self.proc.write(f"{data}\n"), timeout=self.remainingOverageTime + self.time_per_step)
        except asyncio.TimeoutError:
            action, stderr = None, None
        time_used = time.time() - stime

        if stderr != "" and stderr is not None:
            self.log.err(f"stderr:\n{stderr}")

        over_time = time_used - self.time_per_step
        if over_time > 0:
            self.remainingOverageTime -= over_time

        if self.remainingOverageTime <= 0 or action is None:
            self.log.err("timed out...")
            action = None
        else:
            try:
                action = json.loads(action)
                if not isinstance(action, dict):
                    raise ValueError("")
            except:
                self.log.err(f"cannot parse action '{action}'")
                action = None
        return action

