# Lux AI Season 2 Rust Kit

This is a rust implementation inteded for enabling communication and validation
of agent actions, as well as providing some helpful utilities for interacting with
game rules.

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
- Errors over panics
- Executable verification
- Coverage tests

