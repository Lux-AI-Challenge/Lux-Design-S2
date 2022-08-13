import { Replay } from "@/types/replay"

export type LoadReplayAction =
  | LoadObjectReplayOption
  | LoadStringReplayOption
  | LoadFileReplayOption

type LoadObjectReplayOption = {
  type: 'object'
  data: Replay 
}

type LoadStringReplayOption = {
  type: 'string'
  data: string
}

type LoadFileReplayOption = {
  type: 'file'
  data: File
}

/* TODO: not implemented yet */

type LoadKaggleReplayOption = {
  type: 'kaggle'
  data: unknown // TODO
}

type LoadUrlReplayOption = {
  type: 'url'
  data: string // url
}