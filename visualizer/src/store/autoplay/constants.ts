export const initialAutoplay = false

export const SPEEDS = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] as const

export type Speed = (typeof SPEEDS)[number]

export const initialSpeed: Speed = 16

export const speedToIndex: Map<Speed, number> = new Map(SPEEDS.map((x, i) => [x, i]))