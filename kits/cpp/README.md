# Lux AI Season 2 C++ Kit

This is the C++ implementation of the Python kit. It *should* have feature parity.

- [1 File structure](#1-file-structure)
  - [1.1 Code file structure](#11-code-file-structure)
  - [1.2 Adding source files](#12-adding-source-files)
- [2 Building the agent](#2-building-the-agent)
  - [2.1 Locally](#21-locally)
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

For Windows user: It's recommended to set up a Debian or Ubuntu WSL and work inside that, as the convenience
scripts were neither built, nor tested on Windows.

### 2.1 Locally

Requirements:  
- Linux (preferably a Debian based distro so it's close to the Kaggle environment)
- CMake
- make
- curl
- Recent C++ compiler

This kit provides a convenience script to build the agent. By default it will create a build directory named `build`,
initialize the CMake project and run make to build the agent. The resulting binary is `build/agent.out`. This script
can take additional flags to alter some aspects of the process. E.g. disable pedantic warnings or compile in debug mode.

Run `./compile.sh` to run the script. Add `--help` to see available options for it.

### 2.2 Using Docker

Requirements:  
- Docker (if docker requires elevated privileges, then the script has to run with elevated privileges)
- tar

Kaggle will run a compiled binary on their Ubuntu systems. Thus we provide another convenience script which will
create an Ubuntu docker container to build your agent. The compiled content can then be found in a newly created
`docker_build` folder.

Additionally, this script will create a `submission.tar.gz` file containing everything in this folder. This archive
should be used to submit your bot to Kaggle.

## 3 Notes about the code

### 3.1 Observation

The essential type of the code provided inside `lux/` is `lux::Observation`. This type contains everything the agent is
provided with each turn. This includes the board state, units, factories and weather schedule. In addition to that
it also stores the initial configuration so it is able to provide certain convenience functions such as
`getCurrentWeather()`.

*Do not* store member of this type by reference longer than the current turn, because (besides the configuration)
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
- the `lux::Board` contains a `factory_occupancy` map (a 2D array, like the `rubble`) where `factory_occupancy[y][x]` contains either the team id of the team that owns a factory there or -1 if there is no factory
- `lux::Observation` provides `getCurrentWeather()` to get the current `lux::WeatherConfig` as well as `isDay()` to determine if it is day or not
- `lux::Unit` and `lux::Factory` provide functionality to calculate the cost of each action
- `lux::Position` provides a static function `Delta(dir)` which will return a delta position for a given direction

Besides that is also has a few unique features (in `lux/log.hpp` and `lux/exception.hpp`):
- `lux::dumpToJsonFile` will write content of a `json` to an actual JSON file (every step the input is written to `input.json` in the build directory)
- `LUX_LOG` macro can be used to log something (debug build only). It will simply be forwarded to `stderr`.
- `LUX_ASSERT` macro can be used to perform assertions (debug build only). If an assertion fails, it will throw a `lux::Exception`.

