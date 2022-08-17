import type { Replay } from "@/types/replay"
import type { LoadReplayAction } from "./load-replay/types"
import type { Speed, SpeedAction } from "./autoplay/types"
import type { Turn, TurnAction } from "./turn/types"

export type Store = {
  /* the replay file */

  /**
   * the full replay currently loaded, as a JS object.
   * if `null`, indicates that no replay is loaded
   */
  replay: Replay | null


  /* replay loading */

  /**
  * a value in the range [0, 1], indicating the progress in loading the replay. Not necessarily very accurate.
  * if `null`, indicates the replay is not loading (i.e. the replay is either empty, or finished loading).
  */
  progress: number | null
  
  /// similar to a flux dispatch. loads a replay of a given format
  loadReplay: (action: LoadReplayAction) => Promise<void>

  /* replay data */

  /// resets all replay data to their initial state
  resetReplayData: () => void

  /* turn */
  turn: Turn

  /// similar to a flux dispatch. updates the currently viewed turn
  updateTurn: (action: TurnAction) => void
  /* playback speed */

  /// the playback speed
  speed: Speed

  /// similar to a flux dispatch. updates the playback speed
  updateSpeed: (action: SpeedAction) => void


  /* autoplay */

  /// whether the visualizer should uatomatically advance turns
  autoplay: boolean

  toggleAutoplay: (to?: boolean) => void

  /* selected entity */

  /**
   * the currently selected unit or tile.
   * if `null`, indicates no unit or tile selected.
   */
  // TODO
  selectedEntity: any | null

  /* zoom */
  
  tileWidth: number;
}