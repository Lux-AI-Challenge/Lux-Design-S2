import { useStore, useStoreKeys } from "@/store";

import s from "./styles.module.scss";

import groundSvg from "@/assets/ground.svg";
import factory_greenSvg from "@/assets/factory_green.svg";
import factory_redSvg from "@/assets/factory_red.svg";
import { Player } from "@/types/replay/player";
import React, { MouseEventHandler, useEffect, useState } from "react";
import { Bottom } from "./bottom";
import { InteractionLayer } from "@/components/GameMap/interactor";
import { Unit } from "@/types/replay/unit";
import { Store } from "@/store/types";
import { getColor } from "@/utils/colors";
import { Units } from "@/components/GameMap/Units";
interface GameMapProps {
  // hoveredTilePos: {x: number, y: number};
  // setHoveredTilePos: any
  handleOnMouseEnterTile: any;
  viewedTilePos: any;
  handleClickTile: any;
  clickedTilePos: {x: number, y: number} | null;
}
const mapWidth = 48;
const rows = Array.from({ length: mapWidth });
const cols = Array.from({ length: mapWidth });
export const GameMap = React.memo(
  ({
    handleOnMouseEnterTile,
    viewedTilePos,
    handleClickTile,
    clickedTilePos,
  }: GameMapProps) => {
    const replay = useStore((state) => state.replay)!; // game map should only get rendered when replay is non-null
    const { turn, speed, updateGameInfo, tileWidth } = useStoreKeys(
      "turn",
      "speed",
      "updateGameInfo",
      "tileWidth"
    );
    const frame = replay.observations[turn];
    const frameZero = replay.observations[0];
    const mapWidth = frame.board.rubble.length;
    const tileBorder = 1;
    const tileSize = tileWidth + tileBorder * 2;
  
    useEffect(() => {
      // Collect all per turn statistics ahead of time, we should move this out somewhere.
      const posToUnit: Store["gameInfo"]["posToUnit" ]= new Map();
      const posToFactory: Store["gameInfo"]["posToFactory"] = new Map();
      const factoryCounts: Record<string, number> = {};
      const unitCounts: Record<string, number> = {};
      {
        ["player_0", "player_1"].forEach((agent: Player) => {
          factoryCounts[agent] = 0
          Object.entries(frame.factories[agent]).forEach(([factory_id, factory]) => {
            factoryCounts[agent] += 1;
            posToFactory.set(`${factory.pos[0]},${factory.pos[1]}`, factory);
          });
          unitCounts[agent] = Object.keys(frame.units[agent]).length;
          Object.values(frame.units[agent]).forEach((unit) => {
            posToUnit.set(`${unit.pos[0]},${unit.pos[1]}`, unit);
          });
        });
      }
      updateGameInfo({
        type: "set",
        data: { posToUnit, posToFactory, factoryCounts, unitCounts },
      });
    }, [turn]);
    
    // switch dark to day
    let bgColor = "#EF784F"
    if (speed < 200) {
      if (frame.real_env_steps % 50 < 30) {
        // bgColor = "#EF784F"
      } else {
        bgColor = "rgba(0,0,0,0.25)"
      }
    }
    return (
      <>
        <div id="mapContainer" className={s.mapContainer} style={{backgroundColor: bgColor}}>
          {/* bottom layer (height map, rubble, etc) */}
          <Bottom frame={replay.observations[turn]} frameZero={frameZero} />
          {/* top layer (units, buildings, etc) */}
          <div
            className={s.unitLayer}
            style={{
              width: `${tileSize * mapWidth}px`,
            }}
          >
            {["player_0", "player_1"].map((agent: Player) => {
              return Object.values(frame.factories[agent]).map((factory) => {
                let factorySvg = factory_greenSvg;
                if (agent == "player_0") {
                  factorySvg = factory_redSvg;
                }
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
            <Units />
          </div>
          <InteractionLayer
            handleOnMouseEnterTile={handleOnMouseEnterTile}
            handleClickTile={handleClickTile}
            viewedTilePos={viewedTilePos}
            clickedTilePos={clickedTilePos}
          />
        </div>
      </>
    );
  }
);
