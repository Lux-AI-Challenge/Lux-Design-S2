from subprocess import Popen, PIPE
from threading  import Thread
from queue import Queue, Empty

import atexit
import os
import sys
from luxai_runner.logger import Logger
def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()
class BotProcess:
    def __init__(self, command: str, file_path: str, verbose: int = 2) -> None:
        self.command = command
        self.file_path = file_path
        self.log = Logger(verbosity=verbose)
        cwd = os.path.dirname(file_path)
        self.log.info(f"Beginning {self.command} {self.file_path}")
        self._agent_process = Popen([self.command, os.path.basename(self.file_path)], stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=cwd)
        self.log.info(f"Started {self.command} {self.file_path}")
        def cleanup_process():
            self._agent_process.kill()
        atexit.register(cleanup_process)

        # following 4 lines from https://stackoverflow.com/questions/375427/a-non-blocking-read-on-a-subprocess-pipe-in-python
        # used to track stderr data asynchronously
        self._q = Queue()
        self._t = Thread(target=enqueue_output, args=(self._agent_process.stderr, self._q))
        self._t.daemon = True # thread dies with the program
        self._t.start()

    def write(self, msg: str):
        self._agent_process.stdin.write(msg.encode())
        self._agent_process.stdin.flush()
    def receive(self) -> str:
        res = (self._agent_process.stdout.readline()).decode()
        return res

    def print_stderr(self):
        while True:
            try:  line = self._q.get_nowait()
            except Empty:
                # no standard error received, break
                break
            else:
                # standard error output received, print it out
                print(line.decode(), file=sys.stderr, end='')