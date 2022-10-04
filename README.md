# Lux-Design-2022

Welcome to the Lux AI Challenge Season 2! 

The Lux AI Challenge is a competition where competitors design agents to tackle a multi-variable optimization, resource gathering, and allocation problem in a 1v1 scenario against other competitors. In addition to optimization, successful agents must be capable of analyzing their opponents and developing appropriate policies to get the upper hand. 

**We are currently in beta**, so expect unpolished parts of the engine and visuals, bugs, and agent-breaking updates.

To get started, go to our [Getting Started](#getting-started) section. The Beta competition runs until [TODO] and submissions are due at 11:59PM UTC on the competition page: [TODO]

Make sure to join our community discord at https://discord.gg/aWJt3UAcgn to chat, strategize, and learn with other competitors! We will be posting announcements on the Kaggle Forums and on the discord.

Season 2 specifications can be found here: https://lux-ai.org/specs-2022-beta. These detail how the game works and what rules your agent must abide by.

Interested in Season 1? Check out [last year's repository](https://github.com/Lux-AI-Challenge/Lux-Design-2021)

## Getting Started

You will need python 3.8 or above installed on your system. Once installed, you can install the Lux AI season 2 environment with

```
pip install lux-ai-2022
```

To run a match, run


```
python -m luxai_runner.cli my_bot/main.py my_other_bot/main.py -v 2 -o replay.json
```

This will turn on logging to level 2, and store the replay file at `replay.json`. For a full list of commands, type

```
python -m luxai_runner.cli --help
```

Each programming language has a starter kit, you can find general API documentation here: https://github.com/Lux-AI-Challenge/Lux-Design-2022/tree/master/kits

The kits folder in this repository holds all of the available starter kits you can use to start competing and building an AI agent. The readme shows you how to get started with your language of choice and run a match. We strongly recommend reading through the documentation for your language of choice in the links below

Want to use another language but it's not supported? Feel free to suggest that language to our issues or even better, create a starter kit for the community to use and make a PR to this repository. See our CONTRIBUTING.md document for more information on this.

To stay up to date on changes and updates to the competition and the engine, watch for announcements on the forums or the [Discord](https://discord.gg/aWJt3UAcgn). See https://github.com/Lux-AI-Challenge/Lux-Design-2022/blob/master/ChangeLog.md for a full change log.


## Citation
If you use the Lux AI Season 2 environment in your work, please cite this repository as so

@software{Lux_AI_Challenge_S1,
  author = {Tao, Stone and Doerschuk-Tiberi, Bovard},
  month = {9},
  title = {{Lux AI Challenge Season 2}},
  url = {https://github.com/Lux-AI-Challenge/Lux-Design-2022},
  version = {1.0.0},
  year = {2022}
}
