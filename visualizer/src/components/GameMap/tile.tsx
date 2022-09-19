import { useStore, useStoreKeys } from "@/store";

import s from "./styles.module.scss";

import groundSvg from "@/assets/ground.svg";
import factorySvg from "@/assets/factory.svg";
import { Player } from "@/types/replay/player";
import React, { MouseEventHandler, useState } from "react";
import { Frame } from "@/types/replay/frame";

interface BottomProps {
  rubble: number;
  lichen: number;
  lichenStrain: number;
  ice: number;
  ore: number;
  x: number;
  y: number;
  handleOnMouseEnterTile?: any;
}
const mapWidth = 48;
export const GroundTile = React.memo(
  ({ rubble, lichen, ice, ore, x, y }: BottomProps) => {
    const { tileWidth } = useStoreKeys(
      "tileWidth"
    );
    const tileBorder = 1;
    const tileSize = tileWidth + tileBorder * 2;
    if (ice > 0) {
      return (
        <div key={`ice-${y * mapWidth + x}`} className={s.tile}>
          <div
            style={{
              backgroundColor: "#2C9ED3",
              width: tileWidth,
              height: tileWidth,
            }}
          />
        </div>
      );
    }
    if (ore > 0) {
      return (
        <div key={`ore-${y * mapWidth + x}`} className={s.tile}>
          <div
            style={{
              backgroundColor: "#DAA730",
              width: tileWidth,
              height: tileWidth,
            }}
          />
        </div>
      );
    }
    
    let opacity = 0.2 + Math.min(rubble / 100, 1) * 0.8
    let bgColor = "#602009"
    if (rubble == 0) {
      opacity = 0.2;
      bgColor ="#fff"
    }
    // console.log("change")
    return (
      <div
        
        //@ts-ignore
        y={y}
        x={x}
        className={s.tile}
        // onMouseEnter={handleOnMouseEnterTile}
      >
        <div
          id={`lichen-${y * mapWidth + x}`}
          style={{
            position: 'absolute',
            width: tileWidth, height: tileWidth,
            backgroundColor: "#7FCE98",
            willChange: 'opacity',
            opacity: lichen / 10,
          }}
        />
        <div
          id={`rubble-${y * mapWidth + x}`}
          style={{
            width: tileWidth, height: tileWidth,
            backgroundColor: bgColor,
            willChange: 'opacity',
            opacity: opacity,
          }}
        />
      </div>
    );
  }
);
