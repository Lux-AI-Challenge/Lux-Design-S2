import type { Replay } from "@/context/LoadReplayContext/types"

import { useStore, useStoreKeys } from "@/store"

import s from "./styles.module.scss"

import groundSvg from "@/assets/ground.svg"
import factorySvg from "@/assets/factory.svg"

const rows = Array.from({ length: 64 })
const cols = Array.from({ length: 64 })

interface GameMapProps {
}

export function GameMap ({}: GameMapProps) {
  const replay = useStore((state) => state.replay)! // game map should only get rendered when replay is non-null
  const { turn, speed } = useStoreKeys('turn', 'speed')

  const frame = replay.states[turn];
  return (
    <>
      <div className={s.mapContainer}>
        {/* bottom layer (height map, rubble, etc) */}
        <div className={s.mapLayer}>
          {rows.map((_, i) => (
            cols.map((_, j) => (
              <div key={i*cols.length+j} className={s.tile}>
                <img src={groundSvg} width={16} height={16} />
              </div>
            ))
          ))}
        </div>
        
        {/* middle layer (resources, etc) */}
        <div className={s.mapLayer}>
        </div>
        
        {/* top layer (units, buildings, etc) */}
        <div className={s.mapLayer}>
          {Object.values(frame.units.player_0).map((unit) => {
            return (
              <div
                key={unit.unit_id}
                className={s.unit}
                style={{
                  // @ts-ignore
                  "--x": `${unit.pos[0]*16}px`,
                  "--y": `${unit.pos[1]*16}px`,
                  "--t": `calc(1s / ${speed})`, 
                }}
              >
                <img src={factorySvg} width={16} height={16} />
              </div>
            )
          })}
        </div>
      </div>
    </>
  )
}