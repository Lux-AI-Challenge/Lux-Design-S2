import { Charts } from "@/components/InfoPanel/Charts";
import { UnitsList } from "@/components/InfoPanel/UnitsList";
import { TileView } from "@/components/TileView/TileView";
import { WEATHER_ID_TO_NAME, WEATHER_NAME_TO_COLOR } from "@/constants";
import { useStore, useStoreKeys } from "@/store";
import { Replay } from "@/types/replay";
import { Player } from "@/types/replay/player";
import { getColor } from "@/utils/colors";
import React, { useEffect, useState } from "react";

import s from "./styles.module.scss";
type InfoPanelProps = {
  viewedTilePos: { x: number; y: number } | null;
  clickedTilePos: { x: number; y: number } | null;
};

export const InfoPanel = ({
  viewedTilePos,
  clickedTilePos,
}: InfoPanelProps) => {
  const replay: Replay = useStore((state: any) => state.replay)!; // game map should only get rendered when replay is non-null

  const { gameInfo, turn, replayStats } = useStoreKeys(
    "gameInfo",
    "replayStats",
    "turn"
  );
  if (!replay || !replayStats) return <></>;
  const [selectedUnits, setSelectedUnits] = useState<Set<string>>(new Set());
  const frame = replay.observations[turn];
  const frameStats = replayStats.frameStats[turn];
  useEffect(() => {
    let selectedUnits = new Set<string>();
    if (clickedTilePos !== null) {
      // TODO, allow shift clicks
      const unit = gameInfo.posToUnit.get(
        `${clickedTilePos.x},${clickedTilePos.y}`
      );
      const factory = gameInfo.posToFactory.get(
        `${clickedTilePos.x},${clickedTilePos.y}`
      );
      if (unit) {
        selectedUnits.add(unit.unit_id);
      }
      if (factory) {
        selectedUnits.add(factory.unit_id);
      }
    }
    setSelectedUnits(selectedUnits);
  }, [clickedTilePos]);

  // if some ratio is not met, do not display: flex the units lists as they can't fit on a single row
  let flexUnitsTab = true;
  if (window.innerWidth / window.innerHeight < 1.65) {
    flexUnitsTab = false;
  }
  const cur_weather =
    WEATHER_ID_TO_NAME[
      replay.observations[0].weather_schedule[turn - replay.meta.real_start]
    ];
  const get_turn_percent = (turn: number) => {
    return (turn * 100) / (replay.observations.length - replay.meta.real_start);
  };
  const p0_c = getColor("player_0", "");
  const p1_c = getColor("player_1", "");
  return (
    <>
      <div
        className={s.infoPanel}
        //@ts-ignore
        style={{ "--p0_c": p0_c, "--p1_c": p1_c }}
      >
        <div className={s.teams}>
          <div className={s.p0}>
            <p className={s.player}>P0</p>
            <p className={s.teamname}>{replay.meta.teams[0].name}</p>
            {/* <p className={s.factioname}>FACTION NAME</p> */}
          </div>
          <div className={s.p1}>
            <p className={s.player}>P1</p>
            <p className={s.teamname}>{replay.meta.teams[1].name}</p>
            {/* <p className={s.factioname}>FACTION NAME</p> */}
          </div>
        </div>
        <div className={s.body}>
          <div className={s.globalStats}>
            <div className={s.value}>
              <div>{replayStats.frameStats[turn]["player_0"].lichen}</div>
              <div>{gameInfo.factoryCounts["player_0"]}</div>
              <div>{gameInfo.unitCounts["player_0"]}</div>
              <div>{replay.actions[0]["player_0"].bid}</div>
            </div>
            <div>
              <p>Lichen</p>
              <p>Factories</p>
              <p>Robots</p>
              <p>Initial Bid</p>
            </div>
            <div className={s.value}>
              <div>{replayStats.frameStats[turn]["player_1"].lichen}</div>
              <div>{gameInfo.factoryCounts["player_1"]}</div>
              <div>{gameInfo.unitCounts["player_1"]}</div>
              <div>{replay.actions[0]["player_0"].bid}</div>
            </div>
          </div>
          <div
            className={s.liststats}
            style={{ display: flexUnitsTab ? "flex" : "block" }}
          >
            {!flexUnitsTab && <div className={s.liststatHeader}>P0 Units</div>}
            <UnitsList
              flex={flexUnitsTab}
              selectedUnits={selectedUnits}
              frameStats={frameStats["player_0"]}
              units={frame.units["player_0"]}
              factories={frame.factories["player_0"]}
              agent={"player_0"}
            />
            {!flexUnitsTab && <div className={s.liststatHeader}>P1 Units</div>}
            <UnitsList
              flex={flexUnitsTab}
              selectedUnits={selectedUnits}
              frameStats={frameStats["player_1"]}
              units={frame.units["player_1"]}
              factories={frame.factories["player_1"]}
              agent={"player_1"}
            />
          </div>

          <div className={s.weather}>
            <h3 className={s.title}>Weather: {cur_weather}</h3>
            <span
              className={s.slidertick}
              style={{ left: `${get_turn_percent(turn)}%` }}
            ></span>
            <span className={s.slider}></span>
            {replay.meta.weather_events.map((weather) => {
              // TODO add functionlaity for computing start of game excluding place+bid rounds and length of game
              //@ts-ignore
              const styles = {
                left: `${get_turn_percent(weather.start)}%`,
                width: `${get_turn_percent(weather.end - weather.start)}%`,
                backgroundColor: WEATHER_NAME_TO_COLOR[weather.name],
              };
              // if (weather.name == "MARS_QUAKE") {
              //   styles.width = "12px";
              //   styles.left = `calc(${get_turn_percent(weather.start)}% - 6px)`
              //   styles.marginTop = `-4px`;
              //   styles.height = "13px";
              //   styles.borderRadius = "4px";
              // }
              return <span className={s.weatherevent} style={styles}></span>;
            })}
          </div>
          <TileView viewedTilePos={viewedTilePos} />
          {/* <div className={s.chartWrapper}><Charts /></div> */}
        </div>
      </div>
    </>
  );
};
