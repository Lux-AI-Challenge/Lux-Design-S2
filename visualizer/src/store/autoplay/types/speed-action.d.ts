import type { Speed } from "./speed"

export type SpeedAction =
  | ResetSpeedAction
  | IncreaseSpeedAction
  | DecreaseSpeedAction
  | SetSpeedAction

type ResetSpeedAction = {
  type: 'reset'
}

type IncreaseSpeedAction = {
  type: 'increase'
}

type DecreaseSpeedAction = {
  type: 'decrease'
}

type SetSpeedAction = {
  type: 'set'
  data: Speed
}