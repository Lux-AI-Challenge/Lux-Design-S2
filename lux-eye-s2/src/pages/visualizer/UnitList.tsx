import { Text } from '@mantine/core';
import { useEffect, useRef } from 'react';
import { Virtuoso, VirtuosoHandle } from 'react-virtuoso';
import { Tile } from '../../episode/model';
import { useStore } from '../../store';

interface UnitListProps {
  name: string;
  height: number;
  itemCount: number;
  itemRenderer: (index: number) => JSX.Element;
  tileGetter: (index: number) => Tile[];
}

export function UnitList({ name, height, itemCount, tileGetter, itemRenderer }: UnitListProps): JSX.Element {
  const selectedTile = useStore(state => state.selectedTile);

  const ref = useRef<VirtuosoHandle>(null);

  const tiles: Tile[][] = [];
  for (let i = 0; i < itemCount; i++) {
    tiles.push(tileGetter(i));
  }

  useEffect(() => {
    if (selectedTile === null) {
      return;
    }

    const itemIndex = tiles.findIndex(arr => {
      return arr.some(value => value.x === selectedTile.x && value.y === selectedTile.y);
    });

    if (itemIndex > -1) {
      ref.current?.scrollIntoView({ index: itemIndex, behavior: 'smooth' });
    }
  }, [selectedTile]);

  if (itemCount === 0) {
    return (
      <div style={{ height: `${height}px` }}>
        <Text>This team has 0 {name} in this turn.</Text>
      </div>
    );
  }

  return <Virtuoso ref={ref} style={{ height: `${height}px` }} totalCount={itemCount} itemContent={itemRenderer} />;
}
