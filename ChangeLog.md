# ChangeLog

### v1.1.0

Environment:
- Assymmetric Maps are used now
- Bids are now for who goes first, not who gets an extra factory. If bid is tied, player 0 goes first. This early phase is kept parallel for simplicity. Instead, when it is not your turn to place a factory you must skip your turn. Subsequent turns after the starting phase are still in parallel.
- renamed `board.spawns` -> `board.valid_spawns_mask`. Stores the same information (locations on map that you can spawn a factory on). This changes over time based on where new factories are placed.
- Factories can spawn anywhere, but must not overlap any resource or factory.
- Switch to x,y indexing for all map and position related data
- Bumped up initial water/metal resources per factory to 150 each (old: 100)
- Bumped up initial power per factory to 1000 (old: 100)

Python Kits:
- Updated to accept some new observation entires (`teams[player].place_first`, `board.valid_spawns_mask`), and removed old ones (`board.spawns`)
- Added utility function to determine if it is your team's turn to place a factory or not

Bug Fixes:
- Fix bug where move center costs power
- Fix extra line in stderr logging
- Potential fix of Windows issues with verbose error logging
- Fix bug in lichen growth where lichen would grow to new tiles if there is just 1 lichen, it should be 20

### v1.0.6
- Fix bug where game ends at turns < 1000 (kaggle-environments bug)
- Fixed bug with self-destruct actions not being validated or added
- Log unit ids that collided. E.g. `14: 1 Units: (unit_11) collided at 33,44 with [1] unit_9 UnitType.HEAVY at (33, 44) surviving`
- Fixed bug where power costs were recomputed after state change in a single step, causing potentially negative energy in edge cases

### v1.0.5

Environment:
- Fixed bug where it outputs TypeError and can't serialize np.int32. This is caused by windows systems using np.int32 (but utils didn't convert it to a int)
- Fixed bug where factories could overlap
- More informative error message when user submits action instead of action queue
- Fixed bug with weather events potentially overlapping
- Fixed bug where units were not capped by their battery capacity
- Fixed bug where we do not invalidate off map transfers

Visualizer:
- Fixed small offset bug in weather schedule display
### v1.0.4

Initial beta release. Good luck!