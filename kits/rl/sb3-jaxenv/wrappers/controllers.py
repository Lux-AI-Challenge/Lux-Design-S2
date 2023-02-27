import sys
from functools import partial
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from gym import spaces
from jux.actions import FactoryAction, UnitAction, UnitActionType
from jux.env import JuxAction, JuxEnv, JuxEnvBatch
from jux.env import State as JuxState
from jux.unit import UnitType
from luxai_s2.wrappers import Controller


class SimpleUnitDiscreteController(Controller):
    def __init__(self, env: JuxEnv) -> None:
        """
        A simple controller that controls only the robot that will get spawned.
        Moreover, it will always try to spawn one heavy robot if there are none regardless of action given.

        This is a jax based controller. Functions here are defined without the batch dimension, are jax.vmap easily adds the batching

        For the robot unit
        - 4 cardinal direction movement (4 dims)
        - a move center no-op action (1 dim)
        - transfer action just for transferring ice in 4 cardinal directions or center (5)
        - pickup action for power (1 dims)
        - dig action (1 dim)
        - no op action (1 dim) - equivalent to not submitting an action queue which costs power

        It does not include
        - self destruct action
        - recharge action
        - planning (via actions executing multiple times or repeating actions)
        - factory actions
        - transferring power or resources other than ice

        To help understand how to this controller works to map one action space to the original lux action space,
        see how the lux action space is defined in luxai_s2/spaces/action.py

        """
        self.env = env
        self.move_act_dims = 4
        self.transfer_act_dims = 5
        self.pickup_act_dims = 1
        self.dig_act_dims = 1
        self.no_op_dims = 1

        self.move_dim_high = self.move_act_dims
        self.transfer_dim_high = self.move_dim_high + self.transfer_act_dims
        self.pickup_dim_high = self.transfer_dim_high + self.pickup_act_dims
        self.dig_dim_high = self.pickup_dim_high + self.dig_act_dims
        self.no_op_dim_high = self.dig_dim_high + self.no_op_dims

        self.total_act_dims = self.no_op_dim_high
        action_space = spaces.Discrete(self.total_act_dims)
        super().__init__(action_space)

    @partial(jax.jit, static_argnums=(0,))
    def _is_move_action(self, id):
        return id < self.move_dim_high

    @partial(jax.jit, static_argnums=(0,))
    def _get_move_action(self, id):
        # move direction is id + 1 since we don't allow move center here
        return jnp.array([0, id + 1, 0, 0, 0, 1], dtype=jnp.int16)

    @partial(jax.jit, static_argnums=(0,))
    def _is_transfer_action(self, id):
        return id < self.transfer_dim_high

    @partial(jax.jit, static_argnums=(0,))
    def _get_transfer_action(self, id):
        id = id - self.move_dim_high
        transfer_dir = id % 5
        return jnp.array(
            [1, transfer_dir, 0, self.env.env_cfg.max_transfer_amount, 0, 1],
            dtype=jnp.int16,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _is_pickup_action(self, id):
        return id < self.pickup_dim_high

    @partial(jax.jit, static_argnums=(0,))
    def _get_pickup_action(self, id):
        return jnp.array(
            [2, 0, 4, self.env.env_cfg.max_transfer_amount, 0, 1], dtype=jnp.int16
        )

    @partial(jax.jit, static_argnums=(0,))
    def _is_dig_action(self, id):
        return id < self.dig_dim_high

    @partial(jax.jit, static_argnums=(0,))
    def _get_dig_action(self, id):
        return jnp.array([3, 0, 0, 0, 0, 1], dtype=jnp.int16)

    def _is_noop_action(self, id):
        return id >= self.dig_dim_high

    @partial(jax.jit, static_argnums=(0,))
    def _get_noop_action(self, id):
        return jnp.array([0, 0, 0, 0, 0, 0], dtype=jnp.int16)

    @partial(jax.jit, static_argnums=(0,))
    def action_to_jux_action(self, agent: int, state: JuxState, action: npt.NDArray):
        # This controller will make factories spawn one heavy if there are none
        # and generate the appropriate unit action if there is a heavy
        has_at_least_one_unit = state.n_units[agent] > 0
        is_heavy = state.units.unit_type[agent][0] == UnitType.HEAVY
        exists_heavy = has_at_least_one_unit & is_heavy
        jux_action = JuxAction.empty(self.env.env_cfg, self.env.buf_cfg)

        # generate an action for the first heavy unit if it exists
        jux_action: JuxAction = jax.lax.cond(
            exists_heavy,
            lambda: self._gen_unit_jux_action(agent, state, action),
            lambda: jux_action,
        )

        # always spawn a heavy unit if thre isn't one already
        factory_action = jnp.where(
            ~exists_heavy, FactoryAction.BUILD_HEAVY, FactoryAction.DO_NOTHING
        )
        jux_action = jux_action._replace(
            factory_action=jux_action.factory_action.at[agent, 0].set(factory_action)
        )

        return jux_action

    @partial(jax.jit, static_argnums=(0,))
    def _gen_unit_jux_action(
        self, agent: int, state: JuxState, action: npt.NDArray
    ) -> JuxAction:
        # convert in batch
        # note that the first unit is always controlled by actions in the front of an array.
        # create an empty action

        # jux_action.unit_action_queue = UnitAction()
        is_move = self._is_move_action(action)
        is_transfer = self._is_transfer_action(action)
        is_pickup = self._is_pickup_action(action)
        is_dig = self._is_dig_action(action)
        is_noop = self._is_noop_action(action)

        # find the index
        choices = jnp.array([is_move, is_transfer, is_pickup, is_dig, is_noop])
        choice = jnp.argmax(choices)
        action_queue = jax.lax.switch(
            choice,
            [
                self._get_move_action,
                self._get_transfer_action,
                self._get_pickup_action,
                self._get_dig_action,
                lambda x: jnp.zeros(6, dtype=jnp.int16),
            ],
            action,
        )

        jux_action = JuxAction.empty(
            self.env.env_cfg, self.env.buf_cfg
        )  # every leaf is of shape (2, N, 20)
        # TODO - force cast to right types of int8/int16
        jux_action = jux_action._replace(
            unit_action_queue=jux_action.unit_action_queue._replace(
                action_type=jux_action.unit_action_queue.action_type.at[
                    agent, 0, 0
                ].set(action_queue[0]),
                direction=jux_action.unit_action_queue.direction.at[agent, 0, 0].set(
                    action_queue[1]
                ),
                resource_type=jux_action.unit_action_queue.resource_type.at[
                    agent, 0, 0
                ].set(action_queue[2]),
                amount=jux_action.unit_action_queue.amount.at[agent, 0, 0].set(
                    action_queue[3]
                ),
                repeat=jux_action.unit_action_queue.repeat.at[agent, 0, 0].set(
                    action_queue[4]
                ),
                n=jux_action.unit_action_queue.n.at[agent, 0, 0].set(action_queue[5]),
            ),
            unit_action_queue_count=jux_action.unit_action_queue_count.at[agent, 0].set(
                1
            ),
        )
        unit_idx = 0

        # for speed purposes the action queue is stored as a Ring Buffer, with a front and rear and a counter.
        # So an action queue in memory might looks like this
        #         Front     Rear
        #           |        |
        #           v        v
        # [0, 0, Action1, Action2, 0, 0, 0, ....]
        # counter = 2, front = 2, rear = 3
        #
        # If action_queue.count is 0, then action queue is empty.
        # If it is not 0, the first action is stored in index action_queue.front

        # we retrieve the first heavy unit's data by indexing the team (agent), then 0 (unit_idx) for the first unit, then 0 for the first action in the queue
        # note that we use state.units.action_queue.front[agent, 0], which retrieves the index of the first action in the action queue. See above comment for
        # how action queues are stored
        unit_action: UnitAction = jax.tree_map(
            lambda x: x[agent, unit_idx, state.units.action_queue.front[agent, 0]],
            state.units.action_queue.data,
        )
        nonzero_action_queue = ~(state.units.action_queue.count[agent, unit_idx] == 0)
        new_unit_action: UnitAction = jax.tree_map(
            lambda x: x[agent, unit_idx, 0], jux_action.unit_action_queue
        )
        same_action_type = unit_action.action_type == new_unit_action.action_type
        same_dir = unit_action.direction == new_unit_action.direction
        same_amt = unit_action.amount == new_unit_action.amount
        same_r = unit_action.resource_type == new_unit_action.resource_type
        same_n = unit_action.n == new_unit_action.n
        same_rpt = unit_action.repeat == new_unit_action.repeat
        same_action = (
            nonzero_action_queue
            & same_action_type
            & same_dir
            & same_amt
            & same_r
            & same_n
            & same_rpt
        )
        no_op = is_noop | same_action
        jux_action = jux_action._replace(
            unit_action_queue_update=jux_action.unit_action_queue_update.at[
                agent, 0
            ].set(~no_op),
        )
        return jux_action

    def action_masks(self, agent: str, obs: Dict[str, Any]):
        """
        Defines a simplified action mask for this controller's action space

        Doesn't account for whether robot has enough power
        """

        # compute a factory occupancy map that will be useful for checking if a board tile
        # has a factory and which team's factory it is.
        shared_obs = obs[agent]
        factory_occupancy_map = (
            np.ones_like(shared_obs["board"]["rubble"], dtype=int) * -1
        )
        factories = dict()
        for player in shared_obs["factories"]:
            factories[player] = dict()
            for unit_id in shared_obs["factories"][player]:
                f_data = shared_obs["factories"][player][unit_id]
                f_pos = f_data["pos"]
                # store in a 3x3 space around the factory position it's strain id.
                factory_occupancy_map[
                    f_pos[0] - 1 : f_pos[0] + 2, f_pos[1] - 1 : f_pos[1] + 2
                ] = f_data["strain_id"]

        units = shared_obs["units"][agent]
        action_mask = np.zeros((self.total_act_dims), dtype=bool)
        for unit_id in units.keys():
            action_mask = np.zeros(self.total_act_dims)
            # movement is always valid
            action_mask[:4] = True

            # transferring is valid only if the target exists
            unit = units[unit_id]
            pos = np.array(unit["pos"])
            # a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
            move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])
            for i, move_delta in enumerate(move_deltas):
                transfer_pos = np.array(
                    [pos[0] + move_delta[0], pos[1] + move_delta[1]]
                )
                # check if theres a factory tile there
                if (
                    transfer_pos[0] < 0
                    or transfer_pos[1] < 0
                    or transfer_pos[0] >= len(factory_occupancy_map)
                    or transfer_pos[1] >= len(factory_occupancy_map[0])
                ):
                    continue
                factory_there = factory_occupancy_map[transfer_pos[0], transfer_pos[1]]
                if factory_there in shared_obs["teams"][agent]["factory_strains"]:
                    action_mask[
                        self.transfer_dim_high - self.transfer_act_dims + i
                    ] = True

            factory_there = factory_occupancy_map[pos[0], pos[1]]
            on_top_of_factory = (
                factory_there in shared_obs["teams"][agent]["factory_strains"]
            )

            # dig is valid only if on top of tile with rubble or resources or lichen
            board_sum = (
                shared_obs["board"]["ice"][pos[0], pos[1]]
                + shared_obs["board"]["ore"][pos[0], pos[1]]
                + shared_obs["board"]["rubble"][pos[0], pos[1]]
                + shared_obs["board"]["lichen"][pos[0], pos[1]]
            )
            if board_sum > 0 and not on_top_of_factory:
                action_mask[
                    self.dig_dim_high - self.dig_act_dims : self.dig_dim_high
                ] = True

            # pickup is valid only if on top of factory tile
            if on_top_of_factory:
                action_mask[
                    self.pickup_dim_high - self.pickup_act_dims : self.pickup_dim_high
                ] = True
                action_mask[
                    self.dig_dim_high - self.dig_act_dims : self.dig_dim_high
                ] = False

            # no-op is always valid
            action_mask[-1] = True
            break
        return action_mask
