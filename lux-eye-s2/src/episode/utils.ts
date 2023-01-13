import { Factory, Tile } from './model';

export function getFactoryTiles(factory: Factory): Tile[] {
  const tiles: Tile[] = [];
  const center = factory.tile;

  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      tiles.push({
        x: center.x + dx,
        y: center.y + dy,
      });
    }
  }

  return tiles;
}
