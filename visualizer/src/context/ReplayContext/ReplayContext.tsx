import { createContext, useContext, useReducer } from "react"

import type { Nullable } from "@/types/utils"
import { initial, replayReducer } from "./reducer"
import type {
  Replay,
  ReplayContext,
  ReplayProviderProps,
} from "./types"

const replayContext = createContext<ReplayContext | undefined>(undefined)

export function useReplayContext () {
  const data = useContext(replayContext)
  if (data === undefined) {
    throw new Error(`useReplayContext must be used inside a ReplayProvider`)
  }
  return data!
} 

export function ReplayProvider ({ children }: ReplayProviderProps) {
  const [replay, dispatch] = useReducer(replayReducer, initial)

  console.log({replay})
  return (
    <replayContext.Provider value={{ replay, dispatch }}>
      {children}
    </replayContext.Provider>
  )
}

