import { useStoreKeys } from "@/store";
import React from "react";
import s from "./styles.module.scss";
const mapWidth = 64;
const tileWidth = 12;
const rows = Array.from({ length: mapWidth });
const cols = Array.from({ length: mapWidth });
interface InteractionLayerProps {
  // hoveredTilePos: {x: number, y: number};
  // setHoveredTilePos: any
  handleOnMouseEnterTile: any;
  viewedTilePos: { x: number; y: number } | null;
  handleClickTile: any;
}
export const InteractionLayer = React.memo(
  ({
    handleOnMouseEnterTile,
    viewedTilePos,
    handleClickTile,
  }: InteractionLayerProps) => {
    const { tileWidth } = useStoreKeys("tileWidth");
    return (
      <>
        <div
        className={s.mapLayer}
          style={{
            width: `${tileWidth * 64}px`,
            height: `${tileWidth * 64}px`,
            position: 'absolute'
          }}
        >
          {viewedTilePos && (
            <div
              style={{
                width: `${tileWidth}px`,
                height: `${tileWidth}px`,
                border: "1px solid black",
                position: "absolute",
                // top: 0,
                // left: 0,
                transform: `translate3d(${viewedTilePos.x * tileWidth}px, ${
                  viewedTilePos.y * tileWidth
                }px, 0)`,
              }}
            ></div>
          )}
        </div>
        <div
          className={s.mapLayer}
          style={{
            width: `${tileWidth * 64}px`,
            height: `${tileWidth * 64}px`,
            border: "1px solid black",
          }}
          onMouseMove={handleOnMouseEnterTile}
          onClick={handleClickTile}
        ></div>
      </>
    );
  }
);
