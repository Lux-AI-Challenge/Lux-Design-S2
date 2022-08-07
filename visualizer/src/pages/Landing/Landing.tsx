import { useRef, useCallback } from "react"
import { useReplayContext } from "@/context/ReplayContext"
import { parseReplayData } from "@/helpers/replays"

export function Landing () {
  const { replayDispatch: dispatch } = useReplayContext()

  const inputRef = useRef<HTMLInputElement>(null)

  const onButtonClick = useCallback(() => {
    inputRef.current?.click()
  }, [])

  const handleUpload = useCallback(() => {
    const file = inputRef.current?.files?.[0]
    if (!file) { return }
    const name = file.name
    const split = name.split('.')
    const extension = split.at(-1)! // `String.split` always returns at least 1 length array
    if (extension === 'json') {
      file
        .text()
        .then(JSON.parse)
        .then((data) => {
          const parsed = parseReplayData(data)
          dispatch({ type: 'set', replay: parsed })
        })
    }
  }, [])

  return (
    <div>
      {/* title */}
      <h1>Lux AI Season 2 Visualizer</h1>

      {/* upload replay button */}
      <input ref={inputRef} accept=".json, .luxr" type="file" onChange={handleUpload} style={{display: 'none'}} />
      <button onClick={onButtonClick}>upload replay</button>

    </div>
  )
}