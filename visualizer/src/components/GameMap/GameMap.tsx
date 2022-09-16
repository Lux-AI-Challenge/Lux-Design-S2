import { useStore, useStoreKeys } from "@/store";

import s from "./styles.module.scss";

import groundSvg from "@/assets/ground.svg";
import factorySvg from "@/assets/factory.svg";
import { Player } from "@/types/replay/player";
import React, { MouseEventHandler, useEffect, useState } from "react";
import { Bottom } from "./bottom";
import { InteractionLayer } from "@/components/GameMap/interactor";
import { Unit } from "@/types/replay/unit";
interface GameMapProps {
  // hoveredTilePos: {x: number, y: number};
  // setHoveredTilePos: any
  handleOnMouseEnterTile: any;
  viewedTilePos: any;
  handleClickTile: any;
}
const mapWidth = 48;
const rows = Array.from({ length: mapWidth });
const cols = Array.from({ length: mapWidth });
export const GameMap = React.memo(
  ({
    handleOnMouseEnterTile,
    viewedTilePos,
    handleClickTile,
  }: GameMapProps) => {
    const replay = useStore((state) => state.replay)!; // game map should only get rendered when replay is non-null
    const { turn, speed, updateGameInfo, tileWidth } = useStoreKeys(
      "turn",
      "speed",
      "updateGameInfo",
      "tileWidth"
    );
    const frame = replay.states[turn];
    const frameZero = replay.states[0];
    const mapWidth = frame.board.rubble.length;

    // const tileWidth = tileWidth;
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

    // const [dragTranslation, setDragTranslation] = useState({x: 0, y: 0})
    const unitRender: Array<JSX.Element> = [];
    const posToUnit: Map<string, Unit> = new Map();
    const posToFactory: Map<string, Unit> = new Map(); // TODO
    const factoryCounts: Record<string, number> = {};
    const unitCounts: Record<string, number> = {};

    // Collect all statistics ahead of time, we should move this out somewhere.
    {
      ["player_0", "player_1"].forEach((agent: Player) => {
        factoryCounts[agent] = Object.keys(frame.factories[agent]).length;
        unitCounts[agent] = Object.keys(frame.units[agent]).length;
        return Object.values(frame.units[agent]).forEach((unit) => {
          // store units by position
          posToUnit.set(`${unit.pos[0]},${unit.pos[1]}`, unit);
          unitRender.push(
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
      });
    }
    useEffect(() => {
      updateGameInfo({ type: "set", data: { posToUnit, posToFactory, factoryCounts, unitCounts } });
    }, [turn]);
    return (
      <>
        <div id="mapContainer" className={s.mapContainer}>
          {/* bottom layer (height map, rubble, etc) */}
          <Bottom frame={replay.states[1]} frameZero={frameZero} />
          {/* top layer (units, buildings, etc) */}
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
            {unitRender}
          </div>
          <InteractionLayer
            handleOnMouseEnterTile={handleOnMouseEnterTile}
            handleClickTile={handleClickTile}
            viewedTilePos={viewedTilePos}
          />
        </div>
      </>
    );
  }
);
