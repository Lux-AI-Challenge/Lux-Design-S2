export enum RobotType {
  Light,
  Heavy,
}

export enum Resource {
  Ice,
  Ore,
  Water,
  Metal,
  Power,
}

export enum Direction {
  Center,
  Up,
  Right,
  Down,
  Left,
}

export enum Faction {
  None = 'None',
  AlphaStrike = 'AlphaStrike',
  MotherMars = 'MotherMars',
  TheBuilders = 'TheBuilders',
  FirstMars = 'FirstMars',
}

export interface Tile {
  x: number;
  y: number;
}

export interface Cargo {
  ice: number;
  ore: number;
  water: number;
  metal: number;
}

export interface Action {
  type: string;
}

export interface BidAction extends Action {
  type: 'bid';
  bid: number;
  faction: Faction;
}

export interface BuildFactoryAction extends Action {
  type: 'buildFactory';
  center: Tile;
  water: number;
  metal: number;
}

export interface WaitAction extends Action {
  type: 'wait';
}

export interface BuildRobotAction extends Action {
  type: 'buildRobot';
  robotType: RobotType;
}

export interface WaterAction extends Action {
  type: 'water';
}

export interface RepeatableAction extends Action {
  repeat: number;
  n?: number;
}

export interface MoveAction extends RepeatableAction {
  type: 'move';
  direction: Direction;
}

export interface TransferAction extends RepeatableAction {
  type: 'transfer';
  direction: Direction;
  resource: Resource;
  amount: number;
}

export interface PickupAction extends RepeatableAction {
  type: 'pickup';
  resource: Resource;
  amount: number;
}

export interface DigAction extends RepeatableAction {
  type: 'dig';
}

export interface SelfDestructAction extends RepeatableAction {
  type: 'selfDestruct';
}

export interface RechargeAction extends RepeatableAction {
  type: 'recharge';
  targetPower: number;
}

export type SetupAction = BidAction | BuildFactoryAction | WaitAction;
export type FactoryAction = BuildRobotAction | WaterAction;
export type RobotAction = MoveAction | TransferAction | PickupAction | DigAction | SelfDestructAction | RechargeAction;

export interface Board {
  rubble: number[][];
  ore: number[][];
  ice: number[][];
  lichen: number[][];
  strains: number[][];
}

export interface Unit {
  unitId: string;

  tile: Tile;

  power: number;
  cargo: Cargo;
}

export interface Factory extends Unit {
  strain: number;
  action: FactoryAction | null;

  lichen: number;
}

export interface Robot extends Unit {
  type: RobotType;
  actionQueue: RobotAction[];
}

export interface Team {
  name: string;
  faction: Faction;

  water: number;
  metal: number;

  factories: Factory[];
  robots: Robot[];

  strains: Set<number>;

  placeFirst: boolean;
  factoriesToPlace: number;

  action: SetupAction | null;

  error: string | null;
}

export interface Step {
  step: number;
  board: Board;
  teams: [Team, Team];
}

export interface Episode {
  steps: Step[];
}
