import jax
import jax.numpy as jnp


def place_factory_near_random_ice(key: jax.random.KeyArray, player: int, state):
    """
    jax jittable version of the place factory near random ice function

    You can use a non-jittable version as well but unless that function is batched the reset FPS will be low
    """
    valid_spawns_mask = state.board.valid_spawns_mask
    near_ice_positions = (
        jnp.roll(state.board.ice, 2, axis=0)
        | jnp.roll(state.board.ice, -2, axis=0)
        | jnp.roll(state.board.ice, 2, axis=1)
        | jnp.roll(state.board.ice, -2, axis=1)
    )
    spawns_near_ice = valid_spawns_mask & near_ice_positions
    total_candidates = spawns_near_ice.sum()
    key, subkey = jax.random.split(key)
    spawns_near_ice = jnp.argwhere(spawns_near_ice, size=100, fill_value=0)
    spawn_idx = jax.random.randint(subkey, (1,), 0, total_candidates)  # (1, )
    spawn = jnp.array(spawns_near_ice[spawn_idx][0])  # (2, )
    return dict(
        spawn=spawn,
        metal=state.teams.init_metal[player],
        water=state.teams.init_water[player],
    )
