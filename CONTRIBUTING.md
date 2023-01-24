# Lux AI Challenge Season 2 Contributing Guide

If you find a bug or have a feature request, please open an issue [here](https://github.com/Lux-AI-Challenge/Lux-Design-S2/issues) or let us know on our discord, the forums etc.

Want to make an contribution? Create an issue to this repository detailing what you want to change and if approved, make a pull request to this repository with your contribution! If you aren't sure about what to contribute on, check out our [open issues](https://github.com/Lux-AI-Challenge/Lux-Design-S2/issues). Make sure before you start contributing that you fork the repository and work off the main branch.

And of course, please be aware of and read the [Code Of Conduct](https://github.com/Lux-AI-Challenge/Lux-Design-S2/blob/main/CODE_OF_CONDUCT.md)

## Development Setup

For development, we recommend using conda to setup your python environment. Once you have conda (or the faster mamba), clone this repository and setup the environment

```
git clone https://github.com/Lux-AI-Challenge/Lux-Design-S2.git
conda env create -f environment.yml
conda activate luxai_s2
pip install -e luxai_s2
git submodule init
git submodule pull
```

This repository is organized as follows

- luxai_s2/ - all code related to the Season 2 engine and CPU version
- luxai_s2/luxai_runner/ - all code related to running episodes between different bots of different languages
- kits/ - all code for the starter kits that allow competitors to develop strategies and compete
- visualizer/ - the web based visualizer for the competition

## Contributing Starter Kits

If you are interested in contributing a starter kit for a language that is not currently supported, please post an issue about it on our [issue tracker](https://github.com/Lux-AI-Challenge/Lux-Design-S2/issues) so that people do not accidentally do the same things.

Here are a few things to be aware of. The competition servers currently run on Ubuntu 18.04, and has Python, NodeJS, and Java installed on the system, along with a whole ton of other Python packages. Hence, any language that can compile to machine code / binaries on Ubuntu 18.04 can be easily added to the competition.

If you want to get started, we recommend copying the structure of the folder `kits/cpp` or `kits/python`. We require you to provide a README similar to the other kit readmes, along with documentation on how to get started compiling code (if necessary), then running a match using the compiled code. To help understand what the raw observations given by the engine works, see https://github.com/Lux-AI-Challenge/Lux-Design-S2/main/blob/sample_first_obs.json and https://github.com/Lux-AI-Challenge/Lux-Design-S2/main/blob/sample_obs.json for examples.

Moreover, for compiled languages, we recommend also copying over the `create_submission.sh` script and `Dockerfile` in `kits/cpp/` if the language's compiled binaries are OS dependent. For example, for the C++ kit, the dockerfile is used to compile the C++ agent code on Ubuntu 18.04 so then that code can be submitted to the competition servers.

If you have any questions or need help adding a starter kit, the Lux AI team is more than happy to help! Message us anywhere (preferably Github). You can also follow this document https://github.com/themmj/Lux-Design-S2/blob/documentation/kits/starter-kit-notes.md which has some important notes to be aware of when understanding and building a new starter kit.
