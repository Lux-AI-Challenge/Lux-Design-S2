// TODO

import { Frame } from "./frame"
import { Player } from "./player"

// export type Replay = any // unknown
export type KaggleReplay = {
  configuration: any;
  description: any;
  steps: Array<Array<{
    action: any;
    info: any;
    observation: {
      obs: string;
      remainingOverageTime: number;
      player: string;
    };
    reward: number;
    status: string;
  }>>;
}
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
    factoryLichen: Record<string, number>;
    factoryLichenTiles: Record<string, number>;
  }
}

export type ReplayStats = {
  frameStats: FrameStats[];
  mapStats: {
    iceTiles: number;
    oreTiles: number;
  }
}