from turtle import color
from termcolor import colored
import os
TERM_COLORS = os.environ["LUX_COLORS"] == 'False' if "LUX_COLORS" in os.environ else True
class Logger:
    """
    A basic logger

    verbosity - how verbose the logger is
        0 : Log nothing
        1 : Log errors
        2 : Log warnings
        3 : Log everything
    """
    def __init__(self, verbosity: int = 1) -> None:
        self.verbosity = verbosity
    def _print(self, msg: str, color: str):
        if TERM_COLORS:
            print(colored(msg, color))
    def err(self, msg: str):
        if self.verbosity >= 1: self._print(msg, "red")
    def warn(self, msg: str):
        if self.verbosity >= 2: self._print(msg, "orange")
    def info(self, msg: str):
        if self.verbosity >= 3: self._print(msg, "cyan")