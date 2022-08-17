import { Position } from "./position"
import { TeamId } from "./team"

export type UnitId = `unit_${number}`
export type UnitType = "LIGHT" | "HEAVY"
export type Cargo = {
  ice: number;
  ore: number;
  metal: number;
  water: number;
}
export interface Unit {
  action_queue: unknown[]
  cargo: Cargo
  pos: Position
  power: number
  team_id: TeamId
  unit_id: UnitId
  unit_type: UnitType
}

export type FactoryId = `factory_${number}`

export interface Factory {
  cargo: Cargo
  pos: Position
  power: number
  team_id: TeamId
  unit_id: FactoryId
}