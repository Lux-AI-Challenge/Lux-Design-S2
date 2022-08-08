type Grid<T> = T[][]

export type ResourceTile =
  | 'ice'
  | 'lichen'
  | 'lichen_strains'
  | 'ore'
  | 'rubble'

export type ResourcePresence = 0 | 1

export type ResourceMap = Grid<ResourcePresence>