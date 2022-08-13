import type { Resource } from "./resource";

export type Cargo = {
  [K in Resource]: number
}