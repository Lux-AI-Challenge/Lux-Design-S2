// TODO

import { Frame } from "./frame"
import { Player } from "./player"

// export type Replay = any // unknown

export type Replay = {
  observations: Frame[]
  actions: any[] // TODO
}
export type FrameStats = {
  [x: Player]: {
    units: {
      heavy: number, light: number
    },
    lichen: number;
    ice: number;
    water: number;
    ore: number;
    metal: number;
    storedPower: number;
  }
}

export type ReplayStats = {
  frameStats: FrameStats[];
  mapStats: {
    iceTiles: number;
    oreTiles: number;
  }
}