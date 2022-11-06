# ChangeLog

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