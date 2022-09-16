import { useStoreKeys } from "@/store";
import React, { useState } from "react";
import s from "./styles.module.scss";
const mapWidth = 48;
const rows = Array.from({ length: mapWidth });
const cols = Array.from({ length: mapWidth });
interface InteractionLayerProps {
  // hoveredTilePos: {x: number, y: number};
  // setHoveredTilePos: any
  handleOnMouseEnterTile: any;
  viewedTilePos: { x: number; y: number } | null;
  // dragTranslation: { x: number, y:number};
  // setDragTranslation: any;
  handleClickTile: any;
}
export const InteractionLayer = React.memo(
  ({
    handleOnMouseEnterTile,
    viewedTilePos,
    handleClickTile,
    // dragTranslation,
    // setDragTranslation,
  }: InteractionLayerProps) => {
    const [dragTranslation, _setDragTranslation] = useState({x: 0, y: 0})
    const { tileWidth } = useStoreKeys("tileWidth");
    const tileSize = tileWidth + 1 * 2;
    const [mouseDragging, setMouseDragging] = useState(false);
    const [dragTranslationOffset, setDragTranslationOffset] = useState({x: 0, y: 0})
    const setDragTranslation = ({x, y}: any) => {
      const elem = document.getElementById("mapContainer")
      elem!.style.setProperty("transform",`translate(${x}px, ${y}px)`)
      _setDragTranslation({x, y});
    }
    const keyDownHandler = (event: React.KeyboardEvent<HTMLDivElement>) => {
      const speed = 30;
      let newdragTranslation = dragTranslation;
      if (event.code === "KeyW") {
        newdragTranslation = {...dragTranslation, y: dragTranslation.y + speed};
      }
  
      if (event.code === "KeyS") {
        newdragTranslation = {...dragTranslation, y: dragTranslation.y - speed};
      }
      if (event.code === "KeyA") {
        newdragTranslation = {...dragTranslation, x: dragTranslation.x + speed};
      }
      if (event.code === "KeyD") {
        newdragTranslation = {...dragTranslation, x: dragTranslation.x - speed};
      }
      setDragTranslation(newdragTranslation);
  
      // if (event.code === "ArrowLeft") {
      //   setLeft((left) => left - 10);
      // }
  
      // if (event.code === "ArrowRight") {
      //   setLeft((left) => left + 10);
      // }
    };
  
    return (
      <>
        <div
        className={s.mapLayer}
          style={{
            width: `${tileSize * 48}px`,
            height: `${tileSize * 48}px`,
            position: 'absolute'
          }}
        >
          {viewedTilePos && (
            <div
              style={{
                width: `${tileWidth}px`,
                height: `${tileWidth}px`,
                border: "1px solid white",
                position: "absolute",
                transform: `translate3d(${viewedTilePos.x * tileSize}px, ${
                  viewedTilePos.y * tileSize
                }px, 0)`,
              }}
            ></div>
          )}
        </div>
        <div
          tabIndex={0}
          className={s.mapLayer}
          style={{
            width: `${tileSize * 48}px`,
            height: `${tileSize * 48}px`,
            border: "1px solid rgba(0,0,0,0)",
          }}
          onKeyDown={keyDownHandler}
          onMouseMove={(e) => {
            // if (mouseDragging) {
            //   console.log(e.clientX - dragTranslationOffset.x, e.clientY - dragTranslationOffset.y)
            //   setDragTranslation({x:e.clientX - dragTranslationOffset.x, y:e.clientY - dragTranslationOffset.y})
            // }
            handleOnMouseEnterTile(e);
          }}
          onClick={handleClickTile}
          // onMouseDown={(e) => {
          //   // const bounds = e.target.getBoundingClientRect();
          //   // let x = e.clientX - bounds.left;
          //   // let y = e.clientY - bounds.top;
          //   setMouseDragging(true);
          //   setDragTranslationOffset({x: -dragTranslation.x + e.clientX, y: -dragTranslation.y + e.clientY});
          // }}
          // onMouseUp={(e) => {
          //   setDragTranslationOffset({x: e.clientX, y: e.clientY});
          //   setMouseDragging(false);
          // }}
        ></div>
      </>
    );
  }
);
