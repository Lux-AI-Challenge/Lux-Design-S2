# Lux AI Season 2 Rust Kit

This is a rust implementation inteded for enabling communication and validation
of agent actions, as well as providing some helpful utilities for interacting with
game rules.

**Disclaimer**: Although this kit is heavily documented and reasonably tested,
it has been implemented by a third party based on the specification and python
implementation. As such, any deviation from the python implementation with
regards to documentation or behaviour is unintended and should be logged as a bug

## Getting started

For unix users there is a convenience `Makefile` provided for aliasing basic commands.
This has been chosen mainly because there are some slight subtleties for the
agent executable not present in Windows.

This provides `make run`, `make build`, and `make build-dev` for running a
simulation, building a release executable, and building a debug executable respectively

### Running the demo

To get started there is a provided Random agent that randomly places factories,
then creaes heavy robots to gather ice for the factories.

You can run this demo by firstly compiling the runtime by running `cargo build`

Then to run a simulation using just the demo agent you can run the below,
depending on your platform, from the root directory of the kit.

MacOS / linux / unix:

`make run`

Windows:

```bat
luxai-s2 .\target\debug\lux-runtime.exe .\target\debug\lux-runtime.exe --out=replay.json
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
     env_logger::init();
     // swap out agents here
-    let mut agent = RandomAgent::default();
+    let mut agent = MyAgent::default();
     run(&mut agent).unwrap();
 }
```

The runtime provides `env_logger` as its way of using the `log` crate, as such
all logging is recommended to be done through using the macros provided by the `log`
crate

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
.\create_submission.bat
```

Your submission will be available at `./submission.tar.gz`

## Common Pitfalls

- Unix executables must have a `.out` extension to use with the provided runner.
  This is not a requirement for submission builds

