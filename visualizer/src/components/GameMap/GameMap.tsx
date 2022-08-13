import { useStore, useStoreKeys } from "@/store";

import s from "./styles.module.scss";

import groundSvg from "@/assets/ground.svg";
import factorySvg from "@/assets/factory.svg";
import { Player } from "@/types/replay/player";

interface GameMapProps {}

export function GameMap({}: GameMapProps) {
  const replay = useStore((state) => state.replay)!; // game map should only get rendered when replay is non-null
  const { turn, speed } = useStoreKeys("turn", "speed");
  const frame = replay.states[turn];
  const frameZero = replay.states[0];
  const mapWidth = frame.board.rubble.length;
  const rows = Array.from({ length: mapWidth });
  const cols = Array.from({ length: mapWidth });

  const tileWidth = 12;
  const tileBorder = 1;

  const tileSize = tileWidth + tileBorder * 2;
    return (
    <>
      <div className={s.mapContainer}>
        {/* bottom layer (height map, rubble, etc) */}
        <div className={s.mapLayer}>
          {rows.map((_, i) =>
            cols.map((_, j) => {
              if (frameZero.board.ice[i][j] > 0) {
                return (
                  <div key={`ice-${i * cols.length + j}`} className={s.tile}>
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
              if (frameZero.board.ore[i][j] > 0) {
                return (
                  <div key={`ore-${i * cols.length + j}`} className={s.tile}>
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
                <div key={`g-${i * cols.length + j}`} className={s.tile}>
                  <div style={{position: 'absolute', width:tileWidth, height: tileWidth, backgroundColor: "green",
                  opacity: frame.board.lichen[i][j] / 10
                }}>

                  </div>
                  <img
                    src={groundSvg}
                    width={tileWidth}
                    height={tileWidth}
                    style={{
                      opacity: 1 - Math.min(frame.board.rubble[i][j] / 125, 1),
                    }}
                  />
                </div>
              );
            })
          )}
        </div>

        {/* middle layer (resources, etc) */}
        {/* <div className={s.mapLayer}>
        {rows.map((_, i) =>
            cols.map((_, j) => (
              <div key={`ice-${i * cols.length + j}`} className={s.tile}>
                <div  style={{
                  opacity: frame.board.ice[j][i],
                  backgroundColor: 'blue',
                  width:tileWidth,
                  height: tileWidth
                }} />
              </div>
            ))
          )}
        </div> */}

        {/* top layer (units, buildings, etc) */}
        <div
          className={s.unitLayer}
          style={{
            width: `${tileSize * 64}px`,
          }}
        >
          {["player_0", "player_1"].map((agent: Player) => {
            return Object.values(frame.factories[agent]).map((factory) => {
              return (
                <div key={factory.unit_id} className={s.factory} style={{
                  // @ts-ignore
                  "--x": `${factory.pos[0] * tileSize - tileSize + tileBorder}px`,
                  "--y": `${factory.pos[1] * tileSize - tileSize + tileBorder}px`,
                  "--t": `calc(1s / ${speed})`,
                }}>
                  <img src={factorySvg} width={tileSize * 3 - 2 * tileBorder} height={tileSize * 3 - 2*tileBorder} />
                </div>
              )
            })
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
                  <div style={{width:tileWidth, height: tileWidth, borderRadius: "50%", backgroundColor: unit.unit_type === "HEAVY" ? "rgb(112,162,136)" : "rgb(193,215,204)", border: "1px solid black" }}></div>
                </div>
              );
            });
          })}
        </div>
      </div>
    </>
  );
}
