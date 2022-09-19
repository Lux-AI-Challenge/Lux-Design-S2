import { useStore, useStoreKeys } from "@/store";

import s from "./styles.module.scss";

import groundSvg from "@/assets/ground.svg";
import factorySvg from "@/assets/factory.svg";
import { Player } from "@/types/replay/player";
import React, { MouseEventHandler, useEffect, useState } from "react";
import { Bottom } from "./bottom";
import { InteractionLayer } from "@/components/GameMap/interactor";
import { Unit } from "@/types/replay/unit";
import { Store } from "@/store/types";
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
    useEffect(() => {
      // const rows = Array.from({ length: mapWidth });
      // const cols = Array.from({ length: mapWidth });
    }, []);
    const frame = replay.observations[turn];
    const frameZero = replay.observations[0];
    const mapWidth = frame.board.rubble.length;
    const tileBorder = 1;
    const tileSize = tileWidth + tileBorder * 2;

    const [unitRender, setUnitRender] = useState< Array<JSX.Element>>([]);
    
    
    useEffect(() => {
      // Collect all per turn statistics ahead of time, we should move this out somewhere.
      // TODO Cache values? TODO Provide option to load all per turn values for a little smoother replay?
      const posToUnit: Store["gameInfo"]["posToUnit" ]= new Map();
      const posToFactory: Store["gameInfo"]["posToFactory"] = new Map(); // TODO
      const factoryCounts: Record<string, number> = {};
      const unitCounts: Record<string, number> = {};
      // const factoryToLichen: Store["gameInfo"]["factoryToLichen"] = {};
      // const lichen: Store["gameInfo"]["lichen"] = {};
      const playerToFactoryIds: Record<string, Set<string>> = {}
      const turnUnitRender: Array<JSX.Element> = []
      {
        ["player_0", "player_1"].forEach((agent: Player) => {
          factoryCounts[agent] = 0
          playerToFactoryIds[agent] = new Set();
          Object.entries(frame.factories[agent]).forEach(([factory_id, factory]) => {
            playerToFactoryIds[agent].add(factory_id)
            factoryCounts[agent] += 1;
            posToFactory.set(`${factory.pos[0]},${factory.pos[1]}`, factory);
          });
          unitCounts[agent] = Object.keys(frame.units[agent]).length;
          Object.values(frame.units[agent]).forEach((unit) => {
            // store units by position
            posToUnit.set(`${unit.pos[0]},${unit.pos[1]}`, unit);
            turnUnitRender.push(
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
      setUnitRender(turnUnitRender);
      
      // for (let y = 0; y < frame.board.lichen_strains.length; y ++ ) {
      //   for (let x = 0; x < frame.board.lichen_strains[0].length; x ++ ) {
      //     const lichen_strain = frame.board.lichen_strains[y][x];
      //     if 
      //     const factory_id = `factory_${lichen_strain}`;
      //     for (const agent of ["player_0", "player_1"]) {
      //       if (playerToFactoryIds[agent].has(factory_id)) {
      //         break
      //       }
      //   }
      //   }
      // }


      updateGameInfo({
        type: "set",
        data: { posToUnit, posToFactory, factoryCounts, unitCounts },
      });
    }, [turn]);
    let bgColor = "#EF784F"
    // let bgColor = "#373740"
    // console.log({turn, },(turn - 6) % 50 < 30)
    if ((turn - 6) % 50 < 30) {
      
    } else {
      // bgColor = "rgba(0,0,0,0.25)"
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
            clickedTilePos={clickedTilePos}
          />
        </div>
      </>
    );
  }
);
