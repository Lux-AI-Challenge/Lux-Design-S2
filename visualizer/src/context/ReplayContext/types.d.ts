import type { HasChildren } from "src/types/jsx"

export type Replay = unknown

export type ReplayContext = {
  replay: Nullable<Replay>
  dispatch: (action: ReplayAction) => void
}

export interface ReplayProviderProps extends HasChildren {

}

type SetReplayAction = {
  type: 'set'
  replay: Replay
}

type ClearReplayAction = {
  type: 'clear'
}

type ReplayAction = SetReplayAction | ClearReplayAction

