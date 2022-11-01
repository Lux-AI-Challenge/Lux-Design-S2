# Lux-Design-2022

Welcome to the Lux AI Challenge Season 2! 

The Lux AI Challenge is a competition where competitors design agents to tackle a multi-variable optimization, resource gathering, and allocation problem in a 1v1 scenario against other competitors. In addition to optimization, successful agents must be capable of analyzing their opponents and developing appropriate policies to get the upper hand. 

**We are currently in beta**, so expect unpolished parts and bugs of the engine and visuals.

To get started, go to our [Getting Started](#getting-started) section. The Beta competition runs until December 1 and submissions are due at 11:59PM UTC on the competition page: https://www.kaggle.com/c/lux-ai-2022-beta/

Make sure to join our community discord at https://discord.gg/aWJt3UAcgn to chat, strategize, and learn with other competitors! We will be posting announcements on the Kaggle Forums and on the discord.

Season 2 specifications can be found here: https://lux-ai.org/specs-2022-beta. These detail how the game works and what rules your agent must abide by.

Interested in Season 1? Check out [last year's repository](https://github.com/Lux-AI-Challenge/Lux-Design-2021) where we received 22,000+ submissions from 1,100+ teams around the world ranging from scripted agents to Deep Reinforcement Learning.

## Getting Started

You will need python 3.7 or above installed on your system. Once installed, you can install the Lux AI season 2 environment with

```
pip install --upgrade luxai2022
```

To run a match with the CLI tool, run

```
luxai2022 my_bot/main.py my_other_bot/main.py -v 2 -o replay.json
```

This will turn on logging to level 2, and store the replay file at `replay.json`. For documentation on the luxia2022 tool, see https://github.com/Lux-AI-Challenge/Lux-Design-2022/tree/main/luxai_runner/README.md, which includes details on how to run a local tournament to mass evaluate your agents.

Each programming language has a starter kit, you can find general API documentation here: https://github.com/Lux-AI-Challenge/Lux-Design-2022/tree/main/kits

The kits folder in this repository holds all of the available starter kits you can use to start competing and building an AI agent. The readme shows you how to get started with your language of choice and run a match. We strongly recommend reading through the documentation for your language of choice in the links below

- [Python](https://github.com/Lux-AI-Challenge/Lux-Design-2022/tree/main/kits/python/)
- [C++](https://github.com/Lux-AI-Challenge/Lux-Design-2022/tree/main/kits/cpp/)
- Javascript - TBA
- Typescript - TBA
- Java - TBA

Want to use another language but it's not supported? Feel free to suggest that language to our issues or even better, create a starter kit for the community to use and make a PR to this repository. See our [CONTRIBUTING.md](https://github.com/Lux-AI-Challenge/Lux-Design-2022/tree/main/CONTRIBUTING.md) document for more information on this.

To stay up to date on changes and updates to the competition and the engine, watch for announcements on the forums or the [Discord](https://discord.gg/aWJt3UAcgn). See [ChangeLog.md](https://github.com/Lux-AI-Challenge/Lux-Design-2022/blob/main/ChangeLog.md) for a full change log.

## Community Tools
As the community builds tools for the competition, we will post them here!

## Contributing
See the [guide on contributing](https://github.com/Lux-AI-Challenge/Lux-Design-2022/blob/main/CONTRIBUTING.md)

## Sponsors

To be announced at the official release.

## Core Contributors

We like to extend thanks to some of our early core contributors [@duanwilliam](https://github.com/duanwilliam) (Frontend), [@programjames](https://github.com/programjames) (Map generation, Engine optimization), and [@themmj](https://github.com/themmj) (C++ kit, Engine optimization).


## Citation
If you use the Lux AI Season 2 environment in your work, please cite this repository as so

```
@software{Lux_AI_Challenge_S1,
  author = {Tao, Stone and Doerschuk-Tiberi, Bovard},
  month = {10},
  title = {{Lux AI Challenge Season 2}},
  url = {https://github.com/Lux-AI-Challenge/Lux-Design-2022},
  version = {1.0.0},
  year = {2022}
}
```