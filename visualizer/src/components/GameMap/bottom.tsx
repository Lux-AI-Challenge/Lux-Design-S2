import { useStore, useStoreKeys } from "@/store";

import s from "./styles.module.scss";

import groundSvg from "@/assets/ground.svg";
import factorySvg from "@/assets/factory.svg";
import { Player } from "@/types/replay/player";
import React, { MouseEventHandler, useState } from "react";
import { Frame } from "@/types/replay/frame";
import { GroundTile } from "@/components/GameMap/GroundTile";

interface BottomProps {
  // setHoveredTilePos: any;
  frameZero: Frame;
  frame: Frame;
}

export const Bottom = React.memo(
  ({ frame, frameZero }: BottomProps) => {
    const rows = Array.from({ length: 48 });
  const cols = Array.from({ length: 48 });
    return (
      <>   
        {/* bottom layer (height map, rubble, etc) */}
        <div className={s.mapLayer} style={{position: 'relative'}}>
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
