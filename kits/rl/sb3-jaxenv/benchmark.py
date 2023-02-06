import time

import jax
import jax.numpy as jnp
from jux.config import EnvConfig, JuxBufferConfig
from jux.env import JuxEnv
from luxai_s2.wrappers.sb3jax import SB3JaxVecEnv

from wrappers import SimpleUnitDiscreteController, SimpleUnitObservationWrapper

num_envs = 128
N = 256
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
    jux_env, num_envs=num_envs, controller=SimpleUnitDiscreteController(jux_env.env_cfg)
)
stime = time.time()
env.reset(seed=0)
dtime = time.time() - stime
print(f"Compile Time: {dtime:.4f}s")

stime = time.time()
for i in range(N):
    state = env.reset()
dtime = time.time() - stime
print(f"Reset FPS: {(N * num_envs) / dtime :.4f}")