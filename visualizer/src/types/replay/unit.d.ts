import { Position } from "./position"
import { TeamId } from "./team"

export type UnitId = `unit_${number}`

export interface Unit {
  action_queue: unknown[]
  cargo: Cargo
  pos: Position
  power: number
  team_id: TeamId
  unit_id: UnitId
}

export type FactoryId = `factory_${number}`

export interface Factory {
  cargo: Cargo
  pos: Position
  power: number
  team_id: TeamId
  unit_id: FactoryId
}