import { useStore, useStoreKeys } from "@/store";

import s from "./styles.module.scss";

import groundSvg from "@/assets/ground.svg";
import factorySvg from "@/assets/factory.svg";
import { Player } from "@/types/replay/player";
import React, { MouseEventHandler, useState } from "react";
import { Frame } from "@/types/replay/frame";
import { GroundTile } from "@/components/GameMap/tile";

interface BottomProps {
  // setHoveredTilePos: any;
  frameZero: Frame;
  frame: Frame;
}
const rows = Array.from({ length: 64 });
const cols = Array.from({ length: 64 });
export const Bottom = React.memo(
  ({ frame, frameZero }: BottomProps) => {
    const mapWidth = frame.board.rubble.length;
    const { tileWidth } = useStoreKeys(
      "tileWidth"
    );
    const tileBorder = 1;

    const tileSize = tileWidth + tileBorder * 2;

    
    return (
      <>   
        {/* bottom layer (height map, rubble, etc) */}
        <div className={s.mapLayer}>
          {rows.map((_, i) =>
            cols.map((_, j) => {
              return (
                <GroundTile
                  key={`g-${i * cols.length + j}`}
                  rubble={frame.board.rubble[i][j]}
                  ice={frameZero.board.ice[i][j]}
                  ore={frameZero.board.ore[i][j]}
                  lichen={frame.board.lichen[i][j]}
                  lichenStrain={frame.board.lichen_strains[i][j]}
                  x={j}
                  y={i}
                />
              );
              })
          )}
        </div>
      </>
    );
  }
);
