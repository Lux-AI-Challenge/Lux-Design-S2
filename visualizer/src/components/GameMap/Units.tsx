import { useStore, useStoreKeys } from "@/store";

import s from "./styles.module.scss";

import { Player } from "@/types/replay/player";
import React, { useEffect, useState } from "react";
import { getColor } from "@/utils/colors";
interface UnitsProps {
}
export const Units = React.memo(
  ({
  }: UnitsProps) => {
    const replay = useStore((state) => state.replay)!;
    const { turn, speed, tileWidth } = useStoreKeys(
      "turn",
      "speed",
      "tileWidth"
    );
    const frame = replay.observations[turn];
    const tileBorder = 1;
    const tileSize = tileWidth + tileBorder * 2;

    const [unitRender, setUnitRender] = useState< Array<JSX.Element>>([]);
    
    
    useEffect(() => {
      // A optimization is to move this looping code all into one massive loop elsewhere, at the cost of code readability.
      const turnUnitRender: Array<JSX.Element> = []
      {
        ["player_0", "player_1"].forEach((agent: Player) => {
          Object.values(frame.units[agent]).forEach((unit) => {
            let render_shrink = 2;
            const isHeavy =  Math.random() > 0.5;
            if (!isHeavy) {
              render_shrink = 4;
            }
            turnUnitRender.push(
              <div
                key={unit.unit_id}
                className={s.unit}
                style={{
                  // @ts-ignore
                  "--x": `${unit.pos[0] * tileSize + render_shrink / 2}px`,
                  "--y": `${unit.pos[1] * tileSize + render_shrink / 2}px`,
                  "--t": `calc(1s / ${speed})`,
                }}
              >
                <div
                  style={{
                    width: tileWidth - render_shrink,
                    height: tileWidth - render_shrink,
                    borderRadius: isHeavy ? "0" : "50%",
                    backgroundColor: getColor(agent, frame.teams[agent].faction),
                    border: "1px solid white",
                  }}
                ></div>
              </div>
            );
          });
        });
      }
      setUnitRender(turnUnitRender);
    }, [turn]);
    return (
      <>
        {unitRender}
      </>
    );
  }
);
