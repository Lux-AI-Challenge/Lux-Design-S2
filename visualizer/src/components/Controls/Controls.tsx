import { useCallback, useEffect, useMemo, type ComponentProps } from "react"
import Slider from "@mui/base/SliderUnstyled"
import { useStore, useStoreKeys } from "@/store"

import playIcon from "@/assets/generic-icons/play.svg"
import pauseIcon from "@/assets/generic-icons/pause.svg"
import doubleArrowLeftIcon from "@/assets/generic-icons/double-arrow-left.svg"
import doubleArrowRightIcon from "@/assets/generic-icons/double-arrow-right.svg"
import reloadIcon from "@/assets/generic-icons/reload.svg"

import s from "./styles.module.scss"
import { SPEEDS } from "@/store/autoplay/constants"

interface ControlsProps {

}

export function Controls ({}: ControlsProps) {
  const replay = useStore((state) => state.replay)! // controls should only get rendered when replay is non-null
  const {
    turn, updateTurn,
    autoplay, toggleAutoplay,
    speed, updateSpeed,
  } = useStoreKeys('turn', 'updateTurn', 'autoplay', 'toggleAutoplay', 'speed', 'updateSpeed')
  
  const episodeLength = useMemo(() => replay.states.length, [replay.states.length])

  // stop autoplay when reaching the end
  useEffect(() => {
    if (turn === episodeLength - 1) {
      toggleAutoplay(false)
    }
  }, [turn, episodeLength])

  // do the autoplaying
  useEffect(() => {
    if (!autoplay) { return }
    const interval = setInterval(() => {
      updateTurn({ type: 'step' })
    }, 1000 / speed)

    return () => { clearInterval(interval) }
  }, [autoplay, speed])

  const onChangeSliderValue: ComponentProps<typeof Slider>['onChange'] = useCallback((e, val) => {
    updateTurn({ type: 'set', data: val })
    toggleAutoplay(false)
  }, []) // the store's functions are stable references and never change so we don't need them in the dependency array here

  const onClickRestartButton = () => {
    updateTurn({ type: 'reset' })
    toggleAutoplay(false)
  }

  const onClickPlayButton = () => {
    if (turn === episodeLength - 1) { return }
    toggleAutoplay()
  }

  return (
    <>
      <div className={s.controls}>
        <div className={s.turn}>{`Turn ${turn} / ${episodeLength}`}</div>
        <Slider
          className={s.slider}
          value={turn}
          onChange={onChangeSliderValue}
          step={1}
          min={0}
          max={episodeLength - 1} // temporary hardcoded value
        />
        <div className={s.buttons}>
          {/* restart replay button */}
          <button onClick={onClickRestartButton}><img src={reloadIcon} /></button>

          {/* decrease speed button */}
          <button onClick={() => updateSpeed({ type: 'decrease' })} disabled={speed === SPEEDS[0]}>
            <img src={doubleArrowLeftIcon} />
          </button>

          {/* pause/play button */}
          <button onClick={onClickPlayButton}>
            <img src={autoplay ? pauseIcon : playIcon} />
          </button>

          {/* increase speed button */}
          <button onClick={() => updateSpeed({ type: 'increase' })} disabled={speed === SPEEDS[SPEEDS.length - 1]}>
            <img src={doubleArrowRightIcon} />
          </button>

          {/* speed display */}
          <span>{`${speed}x`}</span>
        </div>
      </div>
    </>
  )
}