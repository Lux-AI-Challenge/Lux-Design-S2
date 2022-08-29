import os.path as osp
import os
from subprocess import Popen
from luxai_runner.ext_to_command import ext_to_command
from luxai_runner.process import BotProcess
class Bot:
    def __init__(self, main_file_path: str, agent: str, agent_idx: str, verbose: int = 1) -> None:
        self.main_file_path = main_file_path
        self.file_ext = osp.splitext(self.main_file_path)[-1]
        if self.file_ext not in ext_to_command:
            raise ValueError(f"{self.file_ext} is not a known file extension, we don't know what command to use. \
                Edit https://github.com/Lux-AI-Challenge/Lux-Design-2021/blob/master/luxai_runner/ext_to_command.py \
                    to support a new extension")
        self.command = ext_to_command[self.file_ext]
        self.is_python = self.file_ext == ".py"
        self.agent = agent
        self.agent_idx = agent_idx
        self.verbose = verbose
        self.proc = BotProcess(self.command, self.main_file_path, verbose=verbose)