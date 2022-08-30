from subprocess import Popen, PIPE, STDOUT
from threading  import Thread
from queue import Queue, Empty
import asyncio

import atexit
import os
import sys
from typing import IO
from luxai_runner.logger import Logger
class BotProcess:
    def __init__(self, command: str, file_path: str, verbose: int = 2) -> None:
        self.command = command
        self.file_path = file_path
        self.log = Logger(identifier = "", verbosity=verbose)
    async def start(self):
        cwd = os.path.dirname(self.file_path)
        self.log.info(f"Beginning {self.command} {self.file_path}")
        # self._agent_process = Popen([self.command, os.path.basename(self.file_path)], stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=cwd)
        

        self._agent_process = await asyncio.create_subprocess_exec(self.command, os.path.basename(self.file_path), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=cwd)
        self.log.info(f"Started {self.command} {self.file_path}")
        # def cleanup_process():
        #     self._agent_process.kill()
        # atexit.register(cleanup_process)
        # self._agent_process

        # following 4 lines from https://stackoverflow.com/questions/375427/a-non-blocking-read-on-a-subprocess-pipe-in-python
        # used to track stderr data asynchronously
        # self._q = Queue()
        # self.s = 0
        # def enqueue_output(out: IO, queue: Queue):
        #     lines = []
        #     self.s += 1
            
        #     self.log.err(self.file_path)
        #     # lines = out.readlines(20)
        
        #     for line in out:
        #         lines += [line.decode()]
        #         self.log.err(str(self.s) + " " + lines[-1], end="")
        #         # print(out.seekable())
        #         if not line:
        #             print("BYE")
        #     #     # queue.(line)
        #     self.log.err(lines)
        #     out.close()
        # self._t = Thread(target=enqueue_output, args=(self._agent_process, self._q))
        # self._t.daemon = True # thread dies with the program
        # self._t.start()

    async def write(self, msg: str):
        # self._agent_process.stdin.write(msg.encode())
        stdout, stderr = await self._agent_process.communicate(msg.encode())
        return stdout.decode(), stderr.decode()
        # self._agent_process.stdin.
    async def receive(self) -> str:
        res = (await self._agent_process.stdout.readline()).decode()
        return res

    def stderr(self):
        stderrs = []
        while True:
            try:  line = self._q.get_nowait()
            except Empty:
                # no standard error received, break
                break
            else:
                # standard error output received, print it out
                stderrs.append(line.decode())
        return stderrs
    def print_stderr(self):
        # pass
        while True:
            try:  line = self._q.get_nowait()
            except Empty:
                # no standard error received, break
                break
            else:
                # standard error output received, print it out
                print(line.decode(), file=sys.stderr, end='')