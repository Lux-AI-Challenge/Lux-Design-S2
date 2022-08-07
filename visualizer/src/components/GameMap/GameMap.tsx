import type { Replay } from "@/context/ReplayContext/types"

import s from "./styles.module.scss"

import groundSvg from "@/assets/ground.svg"
import factorySvg from "@/assets/factory.svg"
import { Dispatch, SetStateAction } from "react"

interface GameMapProps {
  replay: Replay
  turn: number
  setTurn: Dispatch<SetStateAction<number>>
}

const rows = Array.from({ length: 64 })
const cols = Array.from({ length: 64 })

export function GameMap ({
  replay,
  turn,
  setTurn,
}: GameMapProps) {
  const frame = replay!.states[turn];

  console.log({turn})
  return (
    <>
      <div className={s.mapContainer}>
        {/* bottom layer (height map, rubble, etc) */}
        <div className={s.mapLayer}>
          {rows.map((_, i) => (
            cols.map((_, j) => (
              <div key={i*cols.length+j}>
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
          {Object.values(frame["units"]["player_0"]).map((unit) => {
            return <div key={unit.unit_id} className={s.lightUnit}
              style={{position: 'absolute', left: 0, top: 0,
                transform: `translate3d(${unit.pos[0]*16}px, ${unit.pos[1]*16}px, 0)`,
                transition: `transform 1s ease`, 
              }}
            >
              <img src={factorySvg} width={16} height={16} />
            </div>
          })}
        </div>
      </div>
    </>
  )
}