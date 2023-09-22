# Lux-Design-S2

[![PyPI version](https://badge.fury.io/py/luxai_s2.svg)](https://badge.fury.io/py/luxai_s2)

Welcome to the Lux AI Challenge Season 2! (Now at NeurIPS 2023)

The Lux AI Challenge is a competition where competitors design agents to tackle a multi-variable optimization, resource gathering, and allocation problem in a 1v1 scenario against other competitors. In addition to optimization, successful agents must be capable of analyzing their opponents and developing appropriate policies to get the upper hand. The goal of the NeurIPS 2023 edition of the competition is to focus on scaling up solutions to maps and game settings larger than the previous competition. 

Key features this season!
- GPU/TPU optimized environment via Jax
- Asymmetric maps and novel mechanics (action efficiency and planning)
- High quality dataset of past episodes of game play from hundreds of human-written agents including the strongest humans have been able to come up with thus far.

Go to our [Getting Started](#getting-started) section to get started programming a bot. The official NeurIPS 2023 competition runs until November 17th and submissions are due at 11:59PM UTC on the competition page: https://www.kaggle.com/competitions/lux-ai-season-2-neurips-stage-2.

Make sure to join our community discord at https://discord.gg/aWJt3UAcgn to chat, strategize, and learn with other competitors! We will be posting announcements on the Kaggle Forums and on the discord.

Environment specifications can be found here: https://lux-ai.org/specs-s2. These detail how the game works and what rules your agent must abide by.

Interested in Season 1? Check out [last year's repository](https://github.com/Lux-AI-Challenge/Lux-Design-2021) where we received 22,000+ submissions from 1,100+ teams around the world ranging from scripted agents to Deep Reinforcement Learning.

 
If you use the Lux AI Season 2 competition/environment in your work, please cite as so

```
@inproceedings{luxais2_neurips_23,
  title         =     {Lux AI Challenge Season 2, NeurIPS Edition},
  author        =     {Stone Tao and Qimai Li and Yuhao Jiang and Jiaxin Chen and Xiaolong Zhu and Bovard Doerschuk-Tiberi and Isabelle Pan and Addison Howard},
  booktitle     =     {Thirty-seventh Conference on Neural Information Processing Systems: Competition Track},
  url           =     {https://github.com/Lux-AI-Challenge/Lux-Design-S2},
  year          =     {2023}
}
```

## Getting Started

You will need Python >=3.8, <3.11  installed on your system. Once installed, you can install the Lux AI season 2 environment and optionally the GPU version with

```
pip install --upgrade luxai_s2
pip install juxai-s2 # installs the GPU version, requires a compatible GPU
```


If you don't know how conda works, I highly recommend setting it up, see the [install instructions](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation). You can then setup the environment as follows

```
conda create -n "luxai_s2" "python==3.9"
conda activate luxai_s2
pip install --upgrade luxai-s2
```


This will install the latest version of the Lux AI Season 2 environment. In particular, the latest versions default game configurations are for the NeurIPS 2023 competition. For those looking for the [competition prior to NeurIPS 2023](https://www.kaggle.com/c/lux-ai-season-2/) (smaller mapsizes and scale), see this [commit](https://github.com/Lux-AI-Challenge/Lux-Design-S2/tree/a96161ad51aaf6ae430b12c14bf81c37ff09dbd7) for code or do `pip install luxai_s2==2.2.0`. 


To verify your installation, you can run the CLI tool by replacing `path/to/bot/main.py` with a path to a bot (e.g. the starter kit in `kits/python/main.py`) and run

```
luxai-s2 path/to/bot/main.py path/to/bot/main.py -v 2 -o replay.json
```

This will turn on logging to level 2, and store the replay file at `replay.json`. For documentation on the luxai-s2 tool, see the [tool's README](https://github.com/Lux-AI-Challenge/Lux-Design-S2/tree/main/luxai_s2/luxai_runner/README.md), which also includes details on how to run a local tournament to mass evaluate your agents. To watch the replay, upload `replay.json` to https://s2vis.lux-ai.org/ (or change `-o replay.json` to `-o replay.html`)

### Starter Kits

Each supported programming language/solution type has its own starter kit, you can find general [API documentation here](https://github.com/Lux-AI-Challenge/Lux-Design-S2/tree/main/kits).

The kits folder in this repository holds all of the available starter kits you can use to start competing and building an AI agent. The readme shows you how to get started with your language of choice and run a match. We strongly recommend reading through the documentation for your language of choice in the links below

- [Python](https://github.com/Lux-AI-Challenge/Lux-Design-S2/tree/main/kits/python/)
- [Reinforcement Learning (Python)](https://github.com/RoboEden/Luxai-s2-Baseline) - This is designed for a lot of customization and suitable for doing your own RL research in this competition
- [Simple Reinforcement Learning tutorial with SB3 (Python)](https://github.com/Lux-AI-Challenge/Lux-Design-S2/tree/main/kits/rl/)
- [C++](https://github.com/Lux-AI-Challenge/Lux-Design-S2/tree/main/kits/cpp/)
- [Javascript](https://github.com/Lux-AI-Challenge/Lux-Design-S2/tree/main/kits/js/)
- [Java](https://github.com/Lux-AI-Challenge/Lux-Design-S2/tree/main/kits/java/)
- [Go](https://github.com/rooklift/golux2/) - (A working bare-bones Go kit)
- Typescript - TBA


Want to use another language but it's not supported? Feel free to suggest that language to our issues or even better, create a starter kit for the community to use and make a PR to this repository. See our [CONTRIBUTING.md](https://github.com/Lux-AI-Challenge/Lux-Design-S2/tree/main/CONTRIBUTING.md) document for more information on this.

If you want to learn how to use the **GPU optimized environment** see [https://github.com/Lux-AI-Challenge/Lux-Design-S2/tree/main/examples/jax_env_tutorial.ipynb](https://github.com/RoboEden/jux/tree/dev)

<!-- For the RL starter kit that trains using the jax env, see https://github.com/Lux-AI-Challenge/Lux-Design-S2/tree/main/kits/rl-sb3-jax-env/ -->

### Episodes Dataset

See https://github.com/RoboEden/Luxai-s2-Baseline for a simple script to download desired episode data from Kaggle. This repository also provides a strong reinforcement learning baseline solution that is easy to iterate and perform research with.

Finally, to stay up to date on changes and updates to the competition and the engine, watch for announcements on the forums or the [Discord](https://discord.gg/aWJt3UAcgn). See [ChangeLog.md](https://github.com/Lux-AI-Challenge/Lux-Design-S2/blob/main/ChangeLog.md) for a full change log.

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

Finally, we are grateful for the support provided by Parametrix.ai in the research and development of this challenge.
