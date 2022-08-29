from typing import Dict
from luxai_runner.bot import Bot


def main():
    pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run LuxAI 2022 game.")
    parser.add_argument('players', nargs="+", help="Paths to player modules.")
    # parser.add_argument("-r", "--rounds", help="Max rounds in game", type=int, default=2000)
    # parser.add_argument("-o", "--output", help="Output file")
    parser.add_argument("-v", "--verbose", help="Verbose Level (0 = silent, 1 = errors, 2 = warnings, 3 = info)", type=int, default=1)

    # # None of these are actually being used yet.
    # parser.add_argument("-t", "--map_type", help="Map type ('Cave', 'Craters', 'Island', 'Mountain')")
    # parser.add_argument("-s", "--size", help="Size (32-64)", type=int)
    # parser.add_argument("-d", "--seed", help="Seed", type=int)
    # parser.add_argument("-m", "--symmetry", help="Symmetry ('horizontal', 'rotational', 'vertical', '/', '\\')")
    args = vars(parser.parse_args())
    if len(args["players"]) != 2:
        raise ValueError("Must provide two paths.")

    players: Dict[str, Bot] = dict()
    for i in range(2):
        player = Bot(args["players"][i], f"player_{i}", i, args["verbose"])
        players[player.agent] = player
    
    for agent in players:
        player = players[agent]
        player.proc.write(f"{player.agent}\n")
        # import ipdb;ipdb.set_trace()
        player.proc.print_stderr()
        data = player.proc.receive()
        print(data)