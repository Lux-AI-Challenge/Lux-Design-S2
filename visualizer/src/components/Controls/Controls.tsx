// https://mui.com/base/react-slider/
import Slider from "@mui/base/SliderUnstyled"
import { ComponentProps, Dispatch, SetStateAction, useCallback } from "react"
import s from "./styles.module.scss"

interface ControlsProps {
  turn: number
  setTurn: Dispatch<SetStateAction<number>>
}

export function Controls ({
  turn,
  setTurn,
}: ControlsProps) {

  const onChangeSliderValue: ComponentProps<typeof Slider>['onChange'] = useCallback((e, val) => {
    setTurn(val)
  }, []) // we don't need `setTurn` in the dependency array here - the returned functions from `useState` are stable references

  return (
    <>
      <div className={s.controls}>
        <div className={s.turn}>Turn {turn}</div>
        <Slider
          className={s.slider}
          value={turn}
          onChange={onChangeSliderValue}
          step={1}
          min={0}
          max={100} // temporary hardcoded value
        />
      </div>
    </>
  )
}