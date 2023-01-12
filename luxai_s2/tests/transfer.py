import numpy as np
from luxai_s2.env import LuxAI_S2

if __name__ == "__main__":
    import time

    env: LuxAI_S2 = LuxAI_S2(verbose=0)
    o = env.reset()
    o, r, d, _ = env.step(
        {
            "player_0": dict(faction="MotherMars", spawns=np.array([[4, 4], [15, 5]])),
            "player_1": dict(
                faction="AlphaStrike", spawns=np.array([[56, 55], [40, 42]])
            ),
        }
    )
