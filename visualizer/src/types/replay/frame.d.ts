import { MapToPlayers } from "./player"
import type { ResourceTile, ResourceMap } from "./resource-map"
import { Factory, FactoryId, Unit, UnitId } from "./unit"

export type Frame = {
  real_env_steps: number;
  board: {
    [K in ResourceTile]: ResourceMap
  }
  team: MapToPlayers<TeamData>
  factories: MapToPlayers<{
    [K: FactoryId]: Factory
  }>
  units: MapToPlayers<{
    [K: UnitId]: Unit
  }>
  weather_schedule: Array<number>;
}