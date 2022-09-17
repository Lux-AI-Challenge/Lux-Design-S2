type Grid<T> = T[][]

export type ResourceTile =
  | 'ice'
  | 'lichen'
  | 'lichen_strains'
  | 'ore'
  | 'rubble'


export type ResourceMap = Grid<number>