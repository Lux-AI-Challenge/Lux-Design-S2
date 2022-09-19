import { useStore, useStoreKeys } from '@/store'
import { Landing } from './pages/Landing'
import { ReplayViewer } from './pages/ReplayViewer'

import './app.css'
import { useEffect, useState } from 'react'

export function App() {
  const replay = useStore((state) => state.replay)
  // for some reason useStore above does not trigger a rerender. Force it here
  // seems to only afflict jupyter notebooks / kaggle
  const [a, setReplay] = useState(null);
  return a === null
    ? <Landing setReplay={setReplay} />
    : <ReplayViewer />
}