from dataclasses import dataclass
from enum import Enum
from typing import Callable

@dataclass
class ActionData:
    type: str
    create: Callable


def move_fn():
    pass
class Action(Enum):
    MOVE = ActionData(type='move', create = move_fn)
    TRANSFER = ActionData(type='transfer')
    PICKUP = ActionData(type='pickup')
    DIG = ActionData(type='dig')
    SELF_DESTRUCT = ActionData(type='self_destruct')



def format_actions():
    # we accept only the structure action tpes, we assume its of shape...
    # TODO
    pass