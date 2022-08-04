import type { Replay } from "@/context/ReplayContext/types"

import s from "./styles.module.scss"

interface GameMapProps {
  replay: Replay
}

export function GameMap ({
  replay,
}: GameMapProps) {
  return (
    <>
      <div className={s.mapContainer}>
        {/* bottom layer (height map, rubble, etc) */}
        <div className={s.mapLayer}>
        </div>
        
        {/* middle layer (resources, etc) */}
        <div className={s.mapLayer}>
        </div>
        
        {/* top layer (units, buildings, etc) */}
        <div className={s.mapLayer}>
        </div>
      </div>
    </>
  )
}