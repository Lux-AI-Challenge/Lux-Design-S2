import asyncio
import json
import os
import os.path as osp
import time
from argparse import Namespace
from subprocess import Popen

from luxai_runner.ext_to_command import ext_to_command
from luxai_runner.logger import Logger
from luxai_runner.process import BotProcess


class Bot:
    def __init__(
        self,
        main_file_path: str,
        agent: str,
        agent_idx: str,
        verbose: int = 1,
        direct_import_python_bots=False,
    ) -> None:
        """
        if direct_import_python_bots is True, will directly import the python agents and call their agent_fn functions.
        """
        self.main_file_path = main_file_path
        self.file_ext = osp.splitext(self.main_file_path)[-1]
        if self.file_ext not in ext_to_command:
            raise ValueError(
                f"{self.file_ext} is not a known file extension, we don't know what command to use. \
                Edit https://github.com/Lux-AI-Challenge/Lux-Design-S2/blob/main/luxai_runner/ext_to_command.py \
                    to support a new extension"
            )
        self.command = ext_to_command[self.file_ext]
        self.is_python = self.file_ext == ".py"
        self.agent = agent
        self.agent_idx = agent_idx
        self.verbose = verbose
        self.direct_import_python_bots = direct_import_python_bots
        self.proc = BotProcess(
            self.command,
            self.main_file_path,
            verbose=verbose,
            direct_import_python_bots=direct_import_python_bots,
        )
        # timing
        self.remainingOverageTime = 60
        self.time_per_step = 3

        self.log = Logger(
            identifier=f"{self.agent}, {self.main_file_path}", verbosity=verbose
        )

    async def step(self, obs, step: int, reward: float = 0, info=dict()):
        stime = time.time()
        import copy

        observations = copy.deepcopy(
            dict(
                obs=obs,
                step=step,
                remainingOverageTime=self.remainingOverageTime,
                player=self.agent,
                reward=float(reward),
                info=info,
            )
        )
        stderr = None
        try:
            if self.direct_import_python_bots and self.command == "python":
                env_cfg = None
                if "env_cfg" in info:
                    env_cfg = observations["info"]["env_cfg"]
                observations = Namespace(**observations)
                observations.obs = json.dumps(observations.obs)
                action = self.proc.agent_fn(observations, dict(env_cfg=env_cfg))
            else:
                data = json.dumps(observations)
                action, stderr = await asyncio.wait_for(
                    self.proc.write(f"{data}\n"),
                    timeout=self.remainingOverageTime + self.time_per_step,
                )
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
                if isinstance(action, dict):
                    return action
                action = json.loads(action)
                if not isinstance(action, dict):
                    raise ValueError("")
            except:
                self.log.err(f"cannot parse action '{action}'")
                action = None
        return action
