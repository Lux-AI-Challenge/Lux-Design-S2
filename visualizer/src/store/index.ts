import create from "zustand";
import shallow from "zustand/shallow";
import {
  loadFromObject,
  loadFromString,
  loadFromFile,
} from "./load-replay/utils";
import { decreaseSpeed, increaseSpeed } from "./autoplay/utils";

import { initial } from "./initial";

import type { Store } from "./types";

import TEMPORARY_REPLAY_FOR_TESTING_ONLY from "@/assets/replay.json";

export const useStore = create<Store>((set, get) => ({
  // replay: initial.replay,
  // TEMPORARY FOR FASTER TESTING ONLY. replace with the above commented out line for the actual app
  replay: loadFromObject(TEMPORARY_REPLAY_FOR_TESTING_ONLY),

  progress: initial.progress,

  clearReplay: () => {
    set({ replay: null, progress: null });
    get().resetReplayData();
  },

  loadReplay: async (action) => {
    switch (action.type) {
      case "object": {
        set({ replay: loadFromObject(action.data), progress: null });
        break;
      }
      case "string": {
        set({ replay: loadFromString(action.data), progress: null });
        break;
      }
      case "file": {
        set({ progress: 0 });
        const replay = await loadFromFile(action.data);
        set({ replay, progress: null });
        break;
      }
      default: {
        
        throw new Error(
          // @ts-expect-error `action.type` should be `never`
          `invalid action type ${action.type} provided to \`loadReplay\`, or \`loadReplay\` doesn't handle all possible action types`
        );
      }
    }
  },

  resetReplayData: () => {
    set({
      turn: initial.turn,
      speed: initial.speed,
    });
  },

  /* current turn in replay being viewed */

  turn: initial.turn,

  updateTurn: (action) => {
    switch (action.type) {
      case "reset": {
        return set({ turn: initial.turn });
      }
      case "step": {
        return set((state) => ({ turn: Math.min(state.turn + action.steps, state.replay!.observations.length - 1) }));
      }
      // TODO: might have to do some stuff to prevent the css transitions from triggering when we do this? idk.
      //  we can probably stop the css transitions by making the transition a css variable and setting it or something? idk
      case "set": {
        return set({ turn: action.data });
      }
      case "move_to": {
        return;
      }
    }
  },

  /* autoplay / playback speed */

  autoplay: initial.autoplay,

  toggleAutoplay: (to) =>
    set((state) => ({
      autoplay: to ?? !state.autoplay,
    })),

  speed: initial.speed,

  updateSpeed: (action) =>
    set((state) => {
      switch (action.type) {
        case "reset":
          return { speed: initial.speed };
        case "increase":
          return { speed: increaseSpeed(state.speed) };
        case "decrease":
          return { speed: decreaseSpeed(state.speed) };
        case "set":
          return { speed: action.data };
      }
    }),

  /* selected entity */
  selectedEntity: initial.selectedEntity,

  tileWidth: initial.tileWidth,

  gameInfo: {
    posToUnit: new Map(),
    posToFactory: new Map(),
    unitCounts: {},
    factoryCounts: {}
  },
  updateGameInfo: (action) =>
    set((state) => {
      switch (action.type) {
        case "set":
          return { gameInfo: action.data };
      }
    }),
}));

/**
 * picks properties of `o` in a list of keys `keys`.
 * ideal DX would spread the keys parameter, but we're only using this in one place (`useStoreKeys`)
 *  so we'll keep `keys` a single array variable to avoid re-spreading the array where we call `pick` in `useStoreKeys`
 */
function pick<O, T extends (keyof O)[]>(o: O, keys: T) {
  return keys.reduce((res, k) => {
    if (k in o) {
      res[k] = o[k];
    }
    return res;
  }, {} as { [K in T[number]]: O[K] });
}

export const useStoreKeys = (...keys: (keyof Store)[]) => {
  return useStore((state) => pick(state, keys), shallow);
};
