import { useStore } from '@/store'
import { Landing } from './pages/Landing'
import { ReplayViewer } from './pages/ReplayViewer'

import './app.css'

export function App() {
  const replay = useStore((state) => state.replay)
  return replay === null
    ? <Landing />
    : <ReplayViewer />
}