import type { HasChildren } from "src/types/jsx"

// TODO
export type Replay = any // unknown

export type ReplayContext = {
  replay: Nullable<Replay>
  replayDispatch: (action: ReplayAction) => void
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

