from jux.config import EnvConfig, JuxBufferConfig
from jux.env import JuxEnv
from luxai_s2.wrappers.sb3jax import SB3JaxVecEnv

from wrappers import SimpleUnitDiscreteController, SimpleUnitObservationWrapper

num_envs = 128
N = 1000
MAX_N_UNITS = 2000
print(
    f"Benching env._upgraded_reset. N={N}, num_envs={num_envs}, MAX_N_UNITS={MAX_N_UNITS}"
)
jux_env = JuxEnv(
    env_cfg=EnvConfig(),
    buf_cfg=JuxBufferConfig(MAX_N_UNITS=MAX_N_UNITS),
)

jux_env.buf_cfg
env = SB3JaxVecEnv(
    jux_env, num_envs=4, controller=SimpleUnitDiscreteController(jux_env.env_cfg)
)
import jax
import jax.numpy as jnp

reset_fn = jax.vmap(
    env._upgraded_reset,
)
import time

stime = time.time()
reset_fn(seed=jnp.arange(num_envs))
dtime = time.time() - stime

print(f"Compile Time: {dtime}s")
stime = time.time()
for i in range(N):
    state = reset_fn(seed=jnp.arange(num_envs) + i * 100)
dtime = time.time() - stime
print(f"Reset FPS: {(N * num_envs) / dtime}")
