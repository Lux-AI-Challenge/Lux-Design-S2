# Lux AI Season 2 Rust Kit

This is a rust implementation inteded for enabling communication and validation
of agent actions, as well as providing some helpful utilities for interacting with
game rules.

**Disclaimer**: Although this kit is heavily documented and well tested. It has
been implemented by a third party based on the specification and python
implementation. As such, any deviation from the python implementation with
regards to documentation or behaviour is unintended and should be logged as a bug

## Implementation Clarifications

- Are factories guaranteed to to have an origin at least `(1,1)` and at most
  `(n-2, m-2)`
- Are the player ids guaranteed to be `player_0`, `player_1` every match, if so
  is it worthwhile calling them a generic player_id or just denoting them the
  same way `LIGHT` and `HEAVY` is denoted
- What's the use of factions / should names outside of the 4 valid values be supported
- I've seen some implementations use BigInteger reprs for bid amounts, is this necessary?
- Is there a meaning to repeating a self destruct action

## Tasks remaining

- Better path finding
- Forward sim?
- Errors over panics
- Executable verification
- Coverage tests
- Integration tests

## Getting started

### Running the demo

To get started there is a provided Random agent that randomly places factories,
whilst creating new heavy robots to gather ice.

You can run this demo by firstly compiling the runtime by running `cargo build`

Then to run a simulation using just the demo agent you can run the below
depending on your platform

MacOS / linux / unix:

```bash
luxai_s2 target/debug/lux-runtime target/debug/lux-runtime --out=replay.json
```

Windows:

```bat
luxai_s2 .\target\debug\lux-runtime.exe .\target\debug\lux-runtime.exe --out=replay.json
```

### Building an agent

It's recommended to implement your agent in the `agent` crate. As long as your
agent implements the `lux::Agent` trait it will be compatible with the runtime.
For ease of use it is recommended to also implement the `Default` trait as the struct
won't be initialised from any state.

To use your agent with the runtime, edit the lux-runtime main file to build
and run your agent

```diff
 fn main() {
     // swap out agents here
-    let mut agent = RandomAgent::default();
+    let mut agent = MyAgent::default();
     run(&mut agent).unwrap();
 }
```

## Create Submission

This kit uses `docker` to build a ubuntu 18.04 compatible binary without having to
worry about installing additional rust toolchains / standard libraries.
To create a submission run the below depending on your platform:

MacOS / linux / unix:

```bash
./create_submission.sh
```

Windows:

```bat
.\create_sbumission.bat
```

Your submission will be available at `./submission.tar.gz`

