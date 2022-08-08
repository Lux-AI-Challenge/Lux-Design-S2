export type Player =
  | 'player_0'
  | 'player_1'

export type MapToPlayers<T> = {
  [K in Player]: T
}