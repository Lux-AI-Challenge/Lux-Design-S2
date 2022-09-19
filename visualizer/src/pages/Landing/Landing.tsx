import { useRef, useCallback } from "react"
import { useStoreKeys } from "@/store"

import uploadIcon from "@/assets/generic-icons/upload.svg"

import s from "./styles.module.scss"
import { Box, CircularProgress } from "@mui/material"

export function Landing () {
  const { progress, loadReplay } = useStoreKeys('progress', 'loadReplay')

  const inputRef = useRef<HTMLInputElement>(null)

  const onButtonClick = useCallback(() => {
    inputRef.current?.click()
  }, [])

  const handleUpload = useCallback(async () => {
    const file = inputRef.current?.files?.[0]
    if (!file) { return }
    const name = file.name
    const split = name.split('.')
    const extension = split.at(-1)! // `String.split` always returns at least 1 length array
    if (extension === 'json') {
      loadReplay({ type: 'file', data: file })
    }
  }, [])

  return (
    <div className={s.root}>
      {/* title */}
      <h1>Lux AI Season 2 Visualizer</h1>

      {/* upload replay button */}
      <input ref={inputRef} accept=".json, .luxr" type="file" onChange={handleUpload} style={{display: 'none'}} />
      <button onClick={onButtonClick} className={s.uploadButton}>
        <img src={uploadIcon} />
        upload replay
      </button>
      {progress !== null && <span><Box sx={{ display: 'flex', }}>Loading... <CircularProgress sx={{color: 'white', ml: '0.5rem'}} size={24}/></Box></span>}
    </div>
  )
}