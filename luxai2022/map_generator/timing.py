import timeit

N = 1000
for map_type in ["cave", "island", "mountain"]:
    time = timeit.timeit(f"{map_type}(64, 64)", setup=f"from generator import {map_type}", number=N)
    print(f"Took {time / N * 1000:.2f}ms for {map_type}.")
