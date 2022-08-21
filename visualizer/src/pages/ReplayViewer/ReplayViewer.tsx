import { Controls } from "@/components/Controls";
import { GameMap } from "@/components/GameMap";
import { TileView } from "@/components/TileView/TileView";
import { useStoreKeys } from "@/store";
import { MouseEventHandler, useState } from "react";

export function ReplayViewer() {
  const { tileWidth } = useStoreKeys("tileWidth");
  const tileSize = tileWidth + 1 * 2;
  const [viewedTilePos, setViewedTilePos] = useState<{
    x: number;
    y: number;
  } | null>(null);
  const [hoveredTilePos, setHoveredTilePos] = useState({ x: 0, y: 0 });
  const [clickedTilePos, setClickedTilePos] = useState<{
    x: number;
    y: number;
  } | null>(null);
  const getTileXYFromEvent = (e: any) => {
    //@ts-ignore
    const bounds = e.target.getBoundingClientRect();
    let x = e.clientX - bounds.left;
    let y = e.clientY - bounds.top;
    x = Math.floor(x / tileSize);
    y = Math.floor(y / tileSize);
    return { x, y };
  };
  const outOfBounds = ({ x, y }: { x: number; y: number }) => {
    return (x < 0 || y < 0 || x >= 64 || y >= 64);
  };
  const handleOnMouseEnterTile: MouseEventHandler<HTMLDivElement> = (e) => {
    const { x, y } = getTileXYFromEvent(e);
    if (outOfBounds({ x, y })) return;
    if (!clickedTilePos) {
      setViewedTilePos({ x, y });
    }
    setHoveredTilePos({ x, y });
  };
  const handleClickTile: MouseEventHandler<HTMLDivElement> = (e) => {
    const { x, y } = getTileXYFromEvent(e);
    if (!clickedTilePos) {
      setClickedTilePos({ x, y });
      setViewedTilePos({ x, y });
    } else if (outOfBounds({ x, y })) {
      return;
    } else if (clickedTilePos.x === x && clickedTilePos.y === y) {
      setClickedTilePos(null);
      setViewedTilePos(null);
    } else {
      // setClickedTilePos({ x, y });
      // setViewedTilePos({ x, y });
      setClickedTilePos(null);
      setViewedTilePos(null);
    }
  };
  return (
    <>
      <GameMap
        handleOnMouseEnterTile={handleOnMouseEnterTile}
        viewedTilePos={viewedTilePos}
        handleClickTile={handleClickTile}
      />
      <Controls />
      <TileView viewedTilePos={viewedTilePos} />
    </>
  );
}
