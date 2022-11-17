# ChangeLog

### v1.1.0

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