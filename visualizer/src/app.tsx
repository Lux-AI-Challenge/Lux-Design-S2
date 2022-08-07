import { ReplayProvider, useReplayContext } from './context/ReplayContext'
import { Landing } from './pages/Landing'
import { ReplayViewer } from './pages/ReplayViewer'

import './app.css'

export function App() {
  return (
    <ReplayProvider>
      <Contents />
    </ReplayProvider>
  )
}

function Contents () {
  const { replay } = useReplayContext()
  return replay === null
    ? <Landing />
    : <ReplayViewer />
}