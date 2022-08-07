import { Controls } from "@/components/Controls";
import { GameMap } from "@/components/GameMap";
import { useReplayContext } from "@/context/ReplayContext";
import { useState } from "react";

export function ReplayViewer () {
  const { replay, replayDispatch: dispatch } = useReplayContext()

  // temporary
  // const maxTurns = replay.steps.length
  
  // TODO: make this a reducer (+ maybe extract this and other parameters to context?) 
  const [turn, setTurn] = useState(0)

  return (
    <>
      <GameMap replay={replay} turn={turn} setTurn={setTurn} />
      <Controls turn={turn} setTurn={setTurn} />
    </>
  )
}