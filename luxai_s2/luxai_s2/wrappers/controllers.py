from abc import ABC

from gym import spaces


class Controller(ABC):
    def __init__(self, action_space: spaces.Space) -> None:
        super().__init__()

    def action_to_lux_action(self, obs, act):
        """
        Takes as input the current "raw observation" and the parameterized action and returns
        an action formatted for the Lux env
        """
        raise NotImplementedError()


class SimpleDiscreteController(Controller):
    def __init__(self, board_width: int, board_height: int) -> None:
        """
        A simple controller that uses a discrete action parameterization for Lux AI S2. It includes

        - 4 cardinal direction movement (4 dims)
        - a move center no-op action (1 dim)
        - transfer action each combination of the (4 cardinal directions plus center) x (resource type or power) (5*5 = 25 dims)
        - self destruct action
        """
        move_act_dims = 5
        transfer_act_dims = 5 * 5
        self_destruct_dims = 1

        total_act_dims = move_act_dims + transfer_act_dims + self_destruct_dims
        action_space = spaces.Box(0, total_act_dims, shape=(board_width, board_height))

        super().__init__(action_space)

    def action_to_lux_action(self, obs, act):
        action = dict()
