import numpy as np
from luxai2022.env import LuxAI2022
if __name__ == "__main__":
    import time

    env: LuxAI2022 = LuxAI2022(verbose=0)
    o = env.reset()
    o, r, d, _ = env.step(
        {
            "player_0": dict(faction="MotherMars", spawns=np.array([[4, 4], [15, 5]])),
            "player_1": dict(faction="AlphaStrike", spawns=np.array([[56, 55], [40, 42]])),
        }
    )