import type { Replay, ReplayStats } from "@/types/replay";
import type { LoadReplayAction } from "./load-replay/types";
import type { Speed, SpeedAction } from "./autoplay/types";
import type { Turn, TurnAction } from "./turn/types";
import { Factory, Unit } from "@/types/replay/unit";

export type Store = {
  /* the replay file */

  /**
   * the full replay currently loaded, as a JS object.
   * if `null`, indicates that no replay is loaded
   */
  replay: Replay | null;

  replayStats: ReplayStats | null;

  /* replay loading */

  /**
   * a value in the range [0, 1], indicating the progress in loading the replay. Not necessarily very accurate.
   * if `null`, indicates the replay is not loading (i.e. the replay is either empty, or finished loading).
   */
  progress: number | null;

  /// similar to a flux dispatch. loads a replay of a given format
  loadReplay: (action: LoadReplayAction) => Promise<void>;

  /* replay data */

  /// resets all replay data to their initial state
  resetReplayData: () => void;

  /* turn */
  turn: Turn;

  /// similar to a flux dispatch. updates the currently viewed turn
  updateTurn: (action: TurnAction) => void;
  /* playback speed */

  /// the playback speed
  speed: Speed;

  /// similar to a flux dispatch. updates the playback speed
  updateSpeed: (action: SpeedAction) => void;

  /* autoplay */

  /// whether the visualizer should uatomatically advance turns
  autoplay: boolean;

  toggleAutoplay: (to?: boolean) => void;

  /* selected entity */

  /**
   * the currently selected unit or tile.
   * if `null`, indicates no unit or tile selected.
   */
  // TODO
  selectedEntity: any | null;

  /* zoom */

  tileWidth: number;

  /* tile info */

  gameInfo: {
    posToUnit: Map<string, Unit>;
    posToFactory: Map<string, Factory>;
    factoryCounts: Record<string, number>;
    unitCounts: Record<string, number>;
    // factoryToLichen: Record<string, {lichen: number; lichenTiles: number;}>;
    // lichen: Record<string, number>;
  };
  // TODO: TYPING
  updateGameInfo: (action: any) => void;
};
