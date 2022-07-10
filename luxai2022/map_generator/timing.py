from generator import *
import cProfile
import pstats

num_runs = 100
import time
t = time.time()
for map_type in ["cave", "craters", "island", "mountain"]:
    exec(f"{map_type}(64, 64)") # for numba caching
    with cProfile.Profile() as pr:
        for i in range(num_runs):
            exec(f"{map_type}(64, 64)")
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(pr).sort_stats(sortby)

    # You can get a lot more info out of this, but this suffices.
    for fcn, profile in ps.get_stats_profile().func_profiles.items():
        if fcn == map_type:
            print(f"{map_type} took {profile.tottime / num_runs * 10000 : .2f} ms.")
