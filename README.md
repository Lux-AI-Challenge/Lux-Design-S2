# Lux-Design-S2

[![PyPI version](https://badge.fury.io/py/luxai_s2.svg)](https://badge.fury.io/py/luxai_s2)

Welcome to the Lux AI Challenge Season 2! 

The Lux AI Challenge is a competition where competitors design agents to tackle a multi-variable optimization, resource gathering, and allocation problem in a 1v1 scenario against other competitors. In addition to optimization, successful agents must be capable of analyzing their opponents and developing appropriate policies to get the upper hand.

Key features this season!
- GPU/TPU optimized environment via Jax
- Asymmetric maps and novel mechanics (action efficiency and planning)
- $55,000 Prize Pool

Go to our [Getting Started](#getting-started) section to get started programming a bot. The official competition runs until April 24th and submissions are due at 11:59PM UTC on the competition page: https://www.kaggle.com/competitions/lux-ai-season-2. There is a **$55,000** prize pool this year thanks to contributions from Kaggle, and our sponsors [QuantCo](https://quantco.com/), [Regression Games](https://www.regression.gg/), and [TSVC](https://tsvcap.com)

Make sure to join our community discord at https://discord.gg/aWJt3UAcgn to chat, strategize, and learn with other competitors! We will be posting announcements on the Kaggle Forums and on the discord.

Season 2 specifications can be found here: https://lux-ai.org/specs-s2. These detail how the game works and what rules your agent must abide by.

Interested in Season 1? Check out [last year's repository](https://github.com/Lux-AI-Challenge/Lux-Design-2021) where we received 22,000+ submissions from 1,100+ teams around the world ranging from scripted agents to Deep Reinforcement Learning.

## Getting Started

You will need Python >=3.7, <3.11  installed on your system. Once installed, you can install the Lux AI season 2 environment and optionally the GPU version with

```
pip install --upgrade luxai_s2
pip install juxai-s2 # installs the GPU version, requires a compatible GPU
```

If you have `gym` installation issues, we recommend running `pip install setuptools==59.8.0`. If you have issues installing `vec-noise`, make sure to read the error output, it's usually because you are missing some C/C++ build tools. If you use conda, we highly recommend creating an environment based on the [environment.yml file in this repo](https://github.com/Lux-AI-Challenge/Lux-Design-S2/blob/main/environment.yml). If you don't know how conda works, I highly recommend setting it up, see the [install instructions](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation).

To create a conda environment and use it run
```
conda env create -f environment.yml
conda activate luxai_s2
```


To verify your installation, you can run the CLI tool by replacing `path/to/bot/main.py` with a path to a bot (e.g. the starter kit in `kits/python/main.py`) and run

```
luxai-s2 path/to/bot/main.py path/to/bot/main.py -v 2 -o replay.json
```

This will turn on logging to level 2, and store the replay file at `replay.json`. For documentation on the luxai-s2 tool, see the [tool's README](https://github.com/Lux-AI-Challenge/Lux-Design-S2/tree/main/luxai_s2/luxai_runner/README.md), which also includes details on how to run a local tournament to mass evaluate your agents. To watch the replay, upload `replay.json` to https://s2vis.lux-ai.org/ (or change `-o replay.json` to `-o replay.html`)

Each supported programming language/solution type has its own starter kit, you can find general [API documentation here](https://github.com/Lux-AI-Challenge/Lux-Design-S2/tree/main/kits).

The kits folder in this repository holds all of the available starter kits you can use to start competing and building an AI agent. The readme shows you how to get started with your language of choice and run a match. We strongly recommend reading through the documentation for your language of choice in the links below

- [Python](https://github.com/Lux-AI-Challenge/Lux-Design-S2/tree/main/kits/python/)
- [Reinforcement Learning (Python)](https://github.com/Lux-AI-Challenge/Lux-Design-S2/tree/main/kits/rl/)
- [C++](https://github.com/Lux-AI-Challenge/Lux-Design-S2/tree/main/kits/cpp/)
- [Javascript](https://github.com/Lux-AI-Challenge/Lux-Design-S2/tree/main/kits/js/)
- [Java](https://github.com/Lux-AI-Challenge/Lux-Design-S2/tree/main/kits/java/)
- [Go](https://github.com/rooklift/golux2/) - (A working bare-bones Go kit)
- Typescript - TBA


Want to use another language but it's not supported? Feel free to suggest that language to our issues or even better, create a starter kit for the community to use and make a PR to this repository. See our [CONTRIBUTING.md](https://github.com/Lux-AI-Challenge/Lux-Design-S2/tree/main/CONTRIBUTING.md) document for more information on this.

<!-- Finally, if you want to learn how to use the GPU optimized env see https://github.com/Lux-AI-Challenge/Lux-Design-S2/tree/main/examples/jax_env_tutorial.ipynb

For the RL starter kit that trains using the jax env, see https://github.com/Lux-AI-Challenge/Lux-Design-S2/tree/main/kits/rl-sb3-jax-env/ -->

To stay up to date on changes and updates to the competition and the engine, watch for announcements on the forums or the [Discord](https://discord.gg/aWJt3UAcgn). See [ChangeLog.md](https://github.com/Lux-AI-Challenge/Lux-Design-S2/blob/main/ChangeLog.md) for a full change log.

## Community Tools
As the community builds tools for the competition, we will post them here!

3rd Party Viewer (This has now been merged into the main repo so check out the lux-eye-s2 folder) - https://github.com/jmerle/lux-eye-2022

## Contributing
See the [guide on contributing](https://github.com/Lux-AI-Challenge/Lux-Design-S2/blob/main/CONTRIBUTING.md)

## Sponsors

We are proud to announce our sponsors [QuantCo](https://quantco.com/), [Regression Games](https://www.regression.gg/), and [TSVC](https://tsvcap.com). They help contribute to the prize pool and provide exciting opportunities to our competitors! For more information about them check out https://www.lux-ai.org/sponsors-s2.

## Core Contributors

We like to extend thanks to some of our early core contributors: [@duanwilliam](https://github.com/duanwilliam) (Frontend), [@programjames](https://github.com/programjames) (Map generation, Engine optimization), and [@themmj](https://github.com/themmj) (C++ kit, Go kit, Engine optimization).

We further like to extend thanks to some of our core contributors during the beta period: [@LeFiz](https://github.com/LeFiz) (Game Design/Architecture), [@jmerle](https://github.com/jmerle) (Visualizer)

We further like to thank the following contributors during the official competition: [@aradite](https://github.com/paradite)(JS Kit), [@MountainOrc](https://github.com/MountainOrc)(Java Kit), [@ArturBloch](https://github.com/ArturBloch)(Java Kit), [@rooklift](https://github.com/rooklift)(Go Kit)


## Citation
If you use the Lux AI Season 2 environment in your work, please cite this repository as so

```
@software{Lux_AI_Challenge_S1,
  author = {Tao, Stone and Doerschuk-Tiberi, Bovard},
  month = {10},
  title = {{Lux AI Challenge Season 2}},
  url = {https://github.com/Lux-AI-Challenge/Lux-Design-S2},
  version = {1.0.0},
  year = {2023}
}
```
