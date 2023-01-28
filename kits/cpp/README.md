# Lux AI Season 2 C++ Kit

This is the C++ implementation of the Python kit. It *should* have feature parity. Please make sure to read the instructions as they are important regarding how you will write a bot and submit it to the competition.

Make sure to check our [Discord](https://discord.gg/aWJt3UAcgn) or the [Kaggle forums](https://www.kaggle.com/c/lux-ai-season-2/discussion) for announcements if there are any breaking changes.

## Getting Started

To get started, download this folder from this repository.

Your core agent code will go into `src/agent.cpp`, and you can create and use more files to help you as well. You should leave `main.py, src/main.cpp` alone as that code enables your agent to compete against other agents on Kaggle.

To quickly test run your agent, first compile your agent by running `./compile.sh` (Linux & MacOS) or `.\compile.bat` (Windows). *It will report any missing dependencies or other errors*. If everything is successful, you should see where the created agent was placed. Then you can run:

Linux & MacOS:
```
luxai_s2 build/agent.out build/agent.out --out=replay.json
```

Windows:
```
luxai_s2 .\build\Release\agent.out.exe .\build\Release\agent.out.exe --out=replay.json
```

This will run the compiled `agent.cpp` code and generate a replay file saved to `replay.json`.

## Developing
Now that you have the code up and running, you are ready to start programming and having some fun!

If you haven't read it already, take a look at the [design specifications for the competition](https://www.lux-ai.org/specs-s2). This will go through the rules and objectives of the competition. For a in-depth tutorial, we provide a jupyter notebook both [locally](https://github.com/Lux-AI-Challenge/Lux-Design-S2/blob/main/kits/python/lux-ai-challenge-season-2-tutorial-python.ipynb) and on [Kaggle](https://www.kaggle.com/code/stonet2000/lux-ai-challenge-season-2-tutorial-python)

All of our kits follow a common API through which you can use to access various functions and properties that will help you develop your strategy and bot. The markdown version is here: https://github.com/Lux-AI-Challenge/Lux-Design-S2/blob/main/kits/README.md, which also describes the observation and action structure/spaces.

## Submitting to Kaggle

Submissions need to be a `.tar.gz` bundle with `main.py` at the top level directory (not nested). To create a submission, first create a binary compiled on Ubuntu (through docker or your computer).

We provide a script to do so, for people working on a OS that is not Ubuntu, `./create_submission.sh` (Linux & MacOS) or `.\create_submission.bat` (Windows).

And if you are running Ubuntu 20.04 natively run

```
./compile.sh -b docker_build
```

to skip using docker.

If it has not been created already, then create a submission.tar.gz file with `tar -czvf submission.tar.gz *`. Upload this under the [My Submissions tab](https://www.kaggle.com/competitions/lux-ai-season-2/submissions) and you should be good to go! Your submission will start with a scheduled game vs itself to ensure everything is working before being entered into the matchmaking pool against the rest of the leaderboard.

## Additional Details

See the rest for more in-depth details regarding the C++ starter kit.

- [1 File structure](#1-file-structure)
  - [1.1 Code file structure](#11-code-file-structure)
  - [1.2 Adding source files](#12-adding-source-files)
- [2 Building the agent](#2-building-the-agent)
  - [2.1 Locally](#21-locally)
    - [2.1.1 Linux & MacOS](#211-linux---macos)
    - [2.1.2 Windows](#212-windows)
  - [2.2 Using Docker](#22-using-docker)
- [3 Notes about the code](#3-notes-about-the-code)
  - [3.1 Observation](#31-observation)
  - [3.2 Agent](#32-agent)
  - [3.3 Actions](#33-actions)
  - [3.4 Additional features](#34-additional-features)


## 1 File structure

This project is structured as a CMake project. This root folder contains setup and convenience functionality. The
actual code resides in the `src` directory.

### 1.1 Code file structure

The code is split into three different parts.

1. `main.cpp`: This is provided by us and should not be changed. It performs JSON (de-)serialization among other things.
2. `lux/`: This is provided by us and should not be changed. It contains all type definition for configuration and observation + their parsing logic.
3. `agent.*`: These files contain an example implementation of a bot. This is where you can make your changes. Do not alter existing function signatures or remove anything from the header. However, you can add additional members and functions to the agent to implement your logic. Of course you are allowed to edit the contents of `setup` and `act`.

### 1.2 Adding source files

Working with a single source file can get fairly cumbersome. You can add additional files and sub-folder in the `src` directory. To include them in the build process, simply add the paths inside the `sources.cmake` file.

## 2 Building the agent

The agent can either be built locally or using a docker container. The former should be used for local testing of the
bot. The latter is intended for submission to Kaggle.

*For Windows user*: Running the `.\compile.bat` script with the `/d` option to create a debug build may sometimes report an error similar to this
```
action.obj : fatal error LNK1163: invalid selection for COMDAT section 0x995 [C:\path\to\Lux-Design-S2\kits\cpp\build\agent.out.vcxproj]
```
This is an issue with the MSVC compiler. Just running it again should do the trick.

Additionally, it was reported that docker may give an error about missing privileges even when running the shell as administrator. This appears to be a known issue and most of the time a reinstall of docker and/or WSL should do the trick.

### 2.1 Locally

Requirements:  
- `curl`
- `tar`
- CMake
- a build system (e.g. `make` or Visual Studio)
- Recent C++ compiler

#### 2.1.1 Linux & MacOS

This kit provides a convenience script to build the agent. By default it will create a build directory named `build`,
initialize the CMake project and run make to build the agent. The resulting binary is `build/agent.out`. This script
can take additional flags to alter some aspects of the process. E.g. disable pedantic warnings or compile in debug mode.

Run `./compile.sh` to run the script. Add `--help` to see available options for it.

#### 2.1.2 Windows

`curl` and `tar` should be preinstalled on Windows.
*If you just want to get started*: The rest of the requirements can be fulfilled by installing the latest Visual Studio Community Edition. It will be used as the build system and provide all other required tools. However, you don't need to use it to edit your code (especially if you are not experienced with C++ development).

This kit provides a convenience script to build the agent. By default it will create a build directory named `build`,
initialize the CMake project and run make to build the agent. The resulting binary is `build\Release\agent.out.exe`. This script
can take additional flags to alter some aspects of the process. E.g. disable pedantic warnings or compile in debug mode.

Run `.\compile.bat` to run the script. Add `/h` to see available options for it.

### 2.2 Using Docker and Submitting to Kaggle

Requirements:  
- Docker (if docker requires elevated privileges, then the script has to run with elevated privileges)
- `tar`

Kaggle will run a compiled binary on their Ubuntu systems. Thus we provide another convenience script
(`./create_submission.sh` for Linux and MacOS or `.\create_submission.bat` for Windows) which will
create an Ubuntu docker container to build your agent. The compiled content can then be found in a newly created
`docker_build` folder.

Additionally, this script will create a `submission.tar.gz` file containing everything in this folder. This archive
should be used to submit your bot to Kaggle in the My Submissions Tab.

## 3 Notes about the code

### 3.1 Observation

The essential type of the code provided inside `lux/` is `lux::Observation`. This type contains everything the agent is
provided with each turn. This includes the board state, units and factories. In addition to that
it also stores the initial configuration so it is able to provide convenience functions.

*Do not* store any member of this type by reference longer than the current turn, because (besides the configuration)
every member may be replaced with new deserialized values the following turn.

### 3.2 Agent

The agent, has a few member variables, which mostly (except for step) correspond to the top level data availble in the JSON:  
- `step`: The current step. After the setup phase, it will contain the value of `real_env_steps`. Thus it will always be >= 0.
- `player`: A string containing the player name (`player_n` where `n` is a number). This can be used to access properties of the Observation, like all units owned by said player.
- `remainingOverageTime`: How much time you have left before making a decision.
- `obs`: This contains the current `lux::Observation`. Through this object you can access everything (e.g. board state, units, etc.).

Note that the lifetime of the agent object lasts for the entire episode. Meaning if you add a member and store
a value in turn `n` then this value will be available in turn `n+1` and following.

The agent provides two functions `setup` and `act`. Both functions simply return json, the following will go into
detail what Lux expects to be returned. Note: You can simply return any type of the `lux` or `std` namespace, because
they can implicitly be converted to the `json` type.

`setup` is called during the first `k` turns. On the very first turn the agent has to return a `lux::BidAction` to
place the initial bet. Every following turn should return a `lux::SpawnAction` to spawn additional factories.

`act` is called after the `k` turns have passed. Then the `step` member of the agent will start with 0 again. This
function should return some form of JSON object (e.g. `std::map` or simply a `json`) which maps from an unit id to
some form of array:
```json
{
    "unit_0": [ ... ],
    "unit_7": [ ... ]
}
```
The array can either contain a `lux::FactoryAction` or `lux::UnitAction` depending on the kind of unit. This will be
used to create an action queue for the particular unit.

### 3.3 Actions

These actions can be created directly using a static function (e.g. `lux::FactoryAction::Water()`) or via convenience
functions provided with the type (e.g. `factory.water(obs)`). The convenience function of each type will use other
convenience functions (e.g. `waterCost(obs)` and `canWater(obs)` to ensure the attempted action is legal (only in
debug build). Thus these convenience functions take an observation as their arguement.

### 3.4 Additional features

Just as the Python kit, this kit also provides some additional information not directly found in the JSON.  
- each unit contains its respective `lux::UnitConfiguration`
- the `lux::Board` contains a `factory_occupancy` map (a 2D array, like the `rubble`) where `factory_occupancy[y][x]` is `true` when a factory occupies the spot at `x` and `y`, false otherwise
- `lux::Observation` provides `isDay()` to determine if it is day or not
- `lux::Unit` and `lux::Factory` provide functionality to calculate the cost of each action
- `lux::Position` provides a static function `Delta(dir)` which will return a delta position for a given direction

Besides that is also has a few unique features intended for debugging. To create a debug build, run `compile.sh` with the `-d` flag.  
- inluding `lux/log.hpp` will give you access to:
    - `lux::dumpToJsonFile` which will write the content of a `json` type variable to an actual JSON file (e.g. in `main.cpp` the input is written to `input.json` in the build directory every step) As of now, it will also do this in non-debug builds.
    - `LUX_LOG` which can be used to log something in debug build only. It will simply be forwarded to `stderr`. In non-debug builds, this statement will not produce any code.
- including `lux/exception.hpp` will give you access to:
    - `LUX_ASSERT` which can be used to perform assertions in debug build only. If an assertion fails, it will throw a `lux::Exception`. In non-debug builds, this statement will not produce any code so don't perform any logic-critical function calls in the assertion expression!

The following is a truncated example of the additional features:
```cpp
#include "lux/log.hpp"
#include "lux/exception.hpp"

// ...

json Agent::act() {
    json actions = json::object();

    // ...

    if (moveCost < unit.power) {
        LUX_ASSERT(unit.power > 0, "unit must have some power at this point");
        // ...
    } else {
        LUX_LOG("unit " << unit.unit_id << " does not have enough power");
    }

    // ...

    lux::dumpToJsonFile("last_actions.json", actions);
    return actions;
}
```

