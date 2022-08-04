import type { Nullable } from "src/types/utils"
import type { Replay, ReplayAction } from "./types"

export const initial = null

export function replayReducer (state: Nullable<Replay>, action: ReplayAction) {
  switch (action.type) {
    case 'clear':
      return initial
    case 'set':
      return action.replay
  }
}