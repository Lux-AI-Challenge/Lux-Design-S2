from jux.state import JuxAction
from jux.state import State as JuxState


def jux_action_to_lux_action(action: JuxAction, state: JuxState):
    # B2B
    return action.to_lux(state)


def jux_state_to_lux_obs(state: JuxState):
    # B2B
    return state.to_lux()


def lux_state_to_jux_state(lux_state, buf_cfg):
    # S2S
    return JuxState.from_lux(lux_state)


from luxai_s2.state import State as LuxState


def lux_obs_to_lux_state(full_obs, env_cfg, n=2) -> LuxState:
    state: LuxState = LuxState.from_obs(full_obs, env_cfg)
    state.board.map.symmetry = "horizontal"
    return state


"""
obs = dict()
state = lux_obs_to_jux_state(obs, buf_cfg)
obs = convert_obs(state)
act = model(obs)
act = jux_action_to_lux_action(act, state)
"""
