import { useStore, useStoreKeys } from "@/store";

import s from "./styles.module.scss";

import groundSvg from "@/assets/ground.svg";
import factorySvg from "@/assets/factory.svg";
import { Player } from "@/types/replay/player";
import React, { MouseEventHandler, useState } from "react";
import { Bottom } from "./bottom";
import { InteractionLayer } from "@/components/GameMap/interactor";
interface GameMapProps {
  // hoveredTilePos: {x: number, y: number};
  // setHoveredTilePos: any
  handleOnMouseEnterTile: any;
  viewedTilePos: any;
  handleClickTile: any;
}
const mapWidth = 64;
const rows = Array.from({ length: mapWidth });
const cols = Array.from({ length: mapWidth });
export const GameMap = React.memo(
  ({ handleOnMouseEnterTile, viewedTilePos, handleClickTile }: GameMapProps) => {
    const replay = useStore((state) => state.replay)!; // game map should only get rendered when replay is non-null
    const { turn, speed } = useStoreKeys("turn", "speed");
    const frame = replay.states[turn];
    const frameZero = replay.states[0];
    const mapWidth = frame.board.rubble.length;

    const tileWidth = 12;
    const tileBorder = 1;

    const tileSize = tileWidth + tileBorder * 2;
    // const factor = 20;
    // for (let i = 0; i < 64; i++) {
    //   for (let j = 0; j < 64; j++) {
    //     const elem = document.getElementById(`lichen-${i*64+j}`)
    //     const rubble = document.getElementById(`rubble-${i*64+j}`)
    //     if (elem) {

    //       const lichen = frame.board.lichen[i][j];
    //       if (!elem.getAttribute("lichen")) {
    //         elem.setAttribute("lichen", `${lichen}`)
    //       }
    //       //@ts-ignore
    //       const oldLichen = parseFloat(elem.getAttribute("lichen"));
    //       // console.log({lichen, oldLichen})
    //     if (Math.round(lichen / factor) != Math.round(oldLichen / factor)) {
    //       elem.style.opacity = `${lichen / 10}`;

    //       }
    //       elem.setAttribute("lichen", `${lichen}`)
    //     }
    //     if (rubble) {
    //       rubble.style.opacity = `${1-Math.min(frame.board.rubble[i][j] / 125, 1)}`
    //     }
    //   }
    // }
    return (
      <>
        <div className={s.mapContainer}>
          
          {/* bottom layer (height map, rubble, etc) */}
            <Bottom frame={replay.states[turn]} frameZero={frameZero} />
          {/* top layer (units, buildings, etc) */}
          <InteractionLayer handleOnMouseEnterTile={handleOnMouseEnterTile} handleClickTile={handleClickTile} viewedTilePos={viewedTilePos} />
          <div
            className={s.unitLayer}
            style={{
              width: `${tileSize * mapWidth}px`,
            }}
          >
            {["player_0", "player_1"].map((agent: Player) => {
              return Object.values(frame.factories[agent]).map((factory) => {
                return (
                  <div
                    key={factory.unit_id}
                    className={s.factory}
                    style={{
                      // @ts-ignore
                      "--x": `${
                        factory.pos[0] * tileSize - tileSize + tileBorder
                      }px`,
                      "--y": `${
                        factory.pos[1] * tileSize - tileSize + tileBorder
                      }px`,
                      "--t": `calc(1s / ${speed})`,
                    }}
                  >
                    <img
                      src={factorySvg}
                      width={tileSize * 3 - 2 * tileBorder}
                      height={tileSize * 3 - 2 * tileBorder}
                    />
                  </div>
                );
              });
            })}
            {["player_0", "player_1"].map((agent: Player) => {
              return Object.values(frame.units[agent]).map((unit) => {
                return (
                  <div
                    key={unit.unit_id}
                    className={s.unit}
                    style={{
                      // @ts-ignore
                      "--x": `${unit.pos[0] * tileSize}px`,
                      "--y": `${unit.pos[1] * tileSize}px`,
                      "--t": `calc(1s / ${speed})`,
                    }}
                  >
                    {/* add back once we have assets */}
                    {/* <img src={factorySvg} width={tileSize} height={tileSize} /> */}
                    <div
                      style={{
                        width: tileWidth,
                        height: tileWidth,
                        borderRadius: "50%",
                        backgroundColor:
                          unit.unit_type === "HEAVY"
                            ? "rgb(112,162,136)"
                            : "rgb(193,215,204)",
                        border: "1px solid black",
                      }}
                    ></div>
                  </div>
                );
              });
            })}
          </div>
        </div>
      </>
    );
  }
);
