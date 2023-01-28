from luxai_s2.globals import TERM_COLORS

try:
    from termcolor import colored
except:
    pass
import os


class Logger:
    """
    A basic logger

    verbosity - how verbose the logger is
        0 : Log nothing
        1 : Log errors
        2 : Log warnings
        3 : Log everything
    """

    def __init__(self, identifier: str, verbosity: int = 1) -> None:
        self.verbosity = verbosity
        self.identifier = identifier

    def _print(self, msg: str, color: str, end: str = "\n"):
        if self.identifier != "":
            msg = f"{self.identifier}: {msg}"
        if TERM_COLORS:
            print(colored(msg, color), end=end)
        else:
            print(msg, end=end)

    def err(self, msg: str, end: str = "\n"):
        if self.verbosity >= 1:
            self._print(msg, "red", end=end)

    def warn(self, msg: str, end: str = "\n"):
        if self.verbosity >= 2:
            self._print(msg, "yellow", end=end)

    def info(self, msg: str, end: str = "\n"):
        if self.verbosity >= 3:
            self._print(msg, "cyan", end=end)
