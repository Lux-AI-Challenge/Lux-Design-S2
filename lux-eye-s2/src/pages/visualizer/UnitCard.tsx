import { Paper } from '@mantine/core';
import { ReactNode, useCallback, useMemo } from 'react';
import { Tile } from '../../episode/model';
import { useStore } from '../../store';

interface UnitCardProps {
  tiles: Tile[];
  tileToSelect: Tile;
  children: ReactNode;
}

export function UnitCard({ tiles, tileToSelect, children }: UnitCardProps): JSX.Element {
  const selectedTile = useStore(state => state.selectedTile);
  const setSelectedTile = useStore(state => state.setSelectedTile);

  const isSelected = tiles.some(tile => selectedTile?.x === tile.x && selectedTile?.y === tile.y);

  const onMouseEnter = useCallback(() => setSelectedTile(tileToSelect, false), [tileToSelect]);
  const onMouseLeave = useCallback(() => setSelectedTile(null, false), []);
  const style = useMemo(() => ({ background: isSelected ? '#ecf0f1' : 'transparent' }), [isSelected]);

  return (
    <div>
      <Paper p="xs" withBorder onMouseEnter={onMouseEnter} onMouseLeave={onMouseLeave} style={style}>
        {children}
      </Paper>
    </div>
  );
}
