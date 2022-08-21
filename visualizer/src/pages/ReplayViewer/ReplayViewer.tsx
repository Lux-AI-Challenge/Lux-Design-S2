import { Controls } from "@/components/Controls";
import { GameMap } from "@/components/GameMap";
import { InfoPanel } from "@/components/InfoPanel";
import { TileView } from "@/components/TileView/TileView";
import { useStoreKeys } from "@/store";
import { MouseEventHandler, useState } from "react";
import s from "./styles.module.scss";
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
    return x < 0 || y < 0 || x >= 64 || y >= 64;
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
      <div className={s.viewer}>
      <div className={s.leftPanel}>
          <InfoPanel viewedTilePos={viewedTilePos} />
        </div>
        <div className={s.gameWrapper}>
          <div className={s.gameMapWrapper}>
          <GameMap
            handleOnMouseEnterTile={handleOnMouseEnterTile}
            viewedTilePos={viewedTilePos}
            handleClickTile={handleClickTile}
          />
          </div>
          <Controls />
        </div>
      </div>
    </>
  );
}
