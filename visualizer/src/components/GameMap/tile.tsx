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
const rows = Array.from({ length: 64 });
const cols = Array.from({ length: 64 });
let i =0;
export const GroundTile = React.memo(
  ({ rubble, lichen, ice, ore, x, y }: BottomProps) => {
    const tileWidth = 12;
    const tileBorder = 1;

    const tileSize = tileWidth + tileBorder * 2;
    if (ice > 0) {
      return (
        <div key={`ice-${y * cols.length + x}`} className={s.tile}>
          <div
            style={{
              backgroundColor: "blue",
              width: tileWidth,
              height: tileWidth,
            }}
          />
        </div>
      );
    }
    if (ore > 0) {
      return (
        <div key={`ore-${y * cols.length + x}`} className={s.tile}>
          <div
            style={{
              backgroundColor: "red",
              width: tileWidth,
              height: tileWidth,
            }}
          />
        </div>
      );
    }

    return (
      <div
        
        //@ts-ignore
        y={y}
        x={x}
        className={s.tile}
        // onMouseEnter={handleOnMouseEnterTile}
      >
        <div
          style={{
            position: "absolute",
            width: tileWidth,
            height: tileWidth,
            backgroundColor: "green",
            opacity: lichen / 10,
          }}
        ></div>
        <img
          src={groundSvg}
          width={tileWidth}
          height={tileWidth}
          style={{
            opacity: 1 - Math.min(rubble / 125, 1),
          }}
        />
      </div>
    );
  },
  (prevProps, nextProps) => {
    // TODO don't change if lichen doesn't enter a new bracket
    // console.log("CHECK", i)
    i += 1;
    if (prevProps.rubble !== nextProps.rubble || prevProps.lichenStrain !== nextProps.lichenStrain || prevProps.handleOnMouseEnterTile !== nextProps.handleOnMouseEnterTile) {
      return false;
    }
    // if (prevProps.lichen !== nextProps.lichen) {
    //   if (nextProps.lichen > 20 && prevProps.lichen < 20) {
    //     return false;
    //   }
    // }
    return true;
  }
);
