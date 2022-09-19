import type { Turn } from "./turn"

export type TurnAction =
  | ResetTurnAction
  | StepTurnAction
  | SetTurnAction
  | MoveToTurnAction

type ResetTurnAction = {
  type: 'reset'
}

type StepTurnAction = {
  type: 'step',
  steps: number
}

type SetTurnAction = {
  type: 'set'
  data: Turn
}

type MoveToTurnAction = {
  type: 'move_to'
  data: Turn
}