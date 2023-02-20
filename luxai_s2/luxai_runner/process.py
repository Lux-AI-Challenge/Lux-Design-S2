import asyncio
import atexit
import os
import sys
from datetime import datetime
from queue import Empty, Queue
from subprocess import PIPE, STDOUT, Popen
from threading import Thread
from typing import IO

from luxai_runner.logger import Logger


class BotProcess:
    def __init__(
        self,
        command: str,
        file_path: str,
        verbose: int = 2,
        live_log: str = True,
        direct_import_python_bots=False,
    ) -> None:
        self.command = command
        if self.command == "./":
            self.is_binary = True
        else:
            self.is_binary = False
        self.file_path = file_path
        self.log = Logger(identifier="", verbosity=verbose)
        self.live_log = live_log
        self.direct_import_python_bots = direct_import_python_bots

        if self.direct_import_python_bots and self.command == "python":
            import importlib.util
            import sys

            sys.path.append(os.path.dirname(file_path))
            spec = importlib.util.spec_from_file_location("bot", file_path)
            foo = importlib.util.module_from_spec(spec)
            sys.modules["bot"] = foo
            spec.loader.exec_module(foo)
            # only use this locally with trusted agents!
            self.agent_fn = foo.agent_fn

    async def start(self):
        if self.direct_import_python_bots and self.command == "python":
            return
        cwd = os.path.dirname(self.file_path)
        if cwd == "":
            cwd = "."
        self.log.info(f"Beginning {self.command} {self.file_path}")
        # self._agent_process = Popen([self.command, os.path.basename(self.file_path)], stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=cwd)

        base_file_path = os.path.basename(self.file_path)
        if self.is_binary:
            self._agent_process = await asyncio.create_subprocess_exec(
                f"{cwd}\{base_file_path}" if sys.platform.startswith('win') else f"./{base_file_path}",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                limit=1024 * 128,
            )
        else:
            if self.command == "java" and base_file_path.endswith(".java"):
                base_file_path = base_file_path[:-5]
            self._agent_process = await asyncio.create_subprocess_exec(
                self.command,
                base_file_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                limit=1024 * 128,
            )
        self.log.info(f"Started {self.command} {self.file_path}")
        # following 4 lines from https://stackoverflow.com/questions/375427/a-non-blocking-read-on-a-subprocess-pipe-in-python
        # used to track stderr data asynchronously
        self._q = Queue()

        def enqueue_output(out: IO, queue: Queue):
            line = out.readline()
            self._q.put(line)

        # self._t = Thread(target=enqueue_output, args=(self._agent_process.stderr, self._q))
        # self._t.daemon = True  # thread dies with the program
        # self._t.start()
        self.stderr_queue = []

        async def log_stream(stream):
            while not stream.at_eof():
                data = await stream.readline()
                line = data.decode()

                if stream.at_eof() and len(line) == 0:
                    break
                elif self.live_log:
                    self.log.err(line, end="")
                else:
                    self.stderr_queue.append(line)

        asyncio.create_task(log_stream(self._agent_process.stderr))
        # await asyncio.gather(watch(self._agent_process.stderr, 'E:'))

    async def write(self, msg: str):
        self._agent_process.stdin.write(msg.encode())
        stdout, stderr = await asyncio.gather(
            self._agent_process.stdout.readline(), self.stderr()
        )
        return stdout.decode(), stderr

    async def receive(self) -> str:
        res = (await self._agent_process.stdout.readline()).decode()
        return res

    async def stderr(self):
        r = "".join(self.stderr_queue)
        self.stderr_queue.clear()
        return r
        # while True:
        #     try:
        #         line = self._q.get_nowait()
        #     except Empty:
        #         # no standard error received, break
        #         break
        #     else:
        #         # standard error output received, print it out
        #         stderrs.append((await line).decode())
        #         # stderrs.append(line.decode())
        # return " ".join(stderrs)

    async def cleanup(self):
        try:
            self._agent_process._transport.close()
            self._agent_process.terminate()
            await self._agent_process.wait()
        except:
            pass
