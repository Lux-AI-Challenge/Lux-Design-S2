import { createContext, useContext, useEffect, useReducer } from "react"

import { initial, replayReducer } from "./reducer"
import type {
  Replay,
  ReplayContext,
  ReplayProviderProps,
} from "./types"

import testReplay from "@/assets/replay.json"

const replayContext = createContext<ReplayContext | undefined>(undefined)

export function useReplayContext () {
  const data = useContext(replayContext)
  if (data === undefined) {
    throw new Error(`useReplayContext must be used inside a ReplayProvider`)
  }
  return data!
} 

export function ReplayProvider ({ children }: ReplayProviderProps) {
  const [replay, replayDispatch] = useReducer(replayReducer, initial)

  // TEMPORARY FOR TESTING, REMOVE LATER
  useEffect(() => {
    replayDispatch({ type: 'set', replay: testReplay })
  }, [])

  return (
    <replayContext.Provider value={{ replay, replayDispatch }}>
      {children}
    </replayContext.Provider>
  )
}