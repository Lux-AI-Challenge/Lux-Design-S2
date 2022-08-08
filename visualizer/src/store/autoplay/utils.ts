import { SPEEDS, speedToIndex } from "./constants";

import type { Speed } from "./types"

export function increaseSpeed (speed: Speed) {
  return SPEEDS[Math.min(SPEEDS.length - 1, speedToIndex.get(speed)! + 1)]
}

export function decreaseSpeed (speed: Speed) {
  return SPEEDS[Math.max(0, speedToIndex.get(speed)! - 1)]
}