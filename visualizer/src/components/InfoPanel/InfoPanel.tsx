import { Charts } from "@/components/InfoPanel/Charts";
import { UnitsList } from "@/components/InfoPanel/UnitsList";
import { TileView } from "@/components/TileView/TileView";
import { useStore, useStoreKeys } from "@/store";
import { Player } from "@/types/replay/player";
import React, { useEffect, useState } from "react";


import s from "./styles.module.scss";
type InfoPanelProps = {
  viewedTilePos: { x: number; y: number } | null;
  clickedTilePos: { x: number; y: number } | null;
}

// move out later
const colors = {
  
}

export const InfoPanel = React.memo(
  ({
    viewedTilePos, clickedTilePos
  }: InfoPanelProps) => {
    const replay = useStore((state: any) => state.replay)!; // game map should only get rendered when replay is non-null
    const { gameInfo, turn, replayStats } = useStoreKeys("gameInfo", "replayStats", "turn");
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
        if (unit) { selectedUnits.add(unit.unit_id)}
        if (factory) { selectedUnits.add(factory.unit_id)}
      }
      setSelectedUnits(selectedUnits);
    }, [clickedTilePos])
    
    return (
      <>
      <div className={s.infoPanel}>
        <div className={s.teams}>
          <div className={s.p0}>
            <p className={s.player}>P0</p>
            <p className={s.teamname}>TEAM NAME</p>
            <p className={s.factioname}>FACTION NAME</p>
          </div>
          <div className={s.p1}>
            <p className={s.player}>P1</p>
            <p className={s.teamname}>TEAM NAME</p>
            <p className={s.factioname}>FACTION NAME</p>
          </div>
        </div>
        <div className={s.body}>
          <div className={s.globalStats}>
            <div className={s.value}>
              <div>{replayStats.frameStats[turn]["player_0"].lichen}</div>
              <div>{gameInfo.factoryCounts["player_0"]}</div>
              <div>{gameInfo.unitCounts["player_0"]}</div>
            </div>
            <div><p>Lichen</p><p>Factories</p><p>Robots</p></div>
            <div className={s.value}>
              <div>{replayStats.frameStats[turn]["player_1"].lichen}</div>
              <div>{gameInfo.factoryCounts["player_1"]}</div>
              <div>{gameInfo.unitCounts["player_1"]}</div>
            </div>
          </div>
          <div className={s.liststats}>
            <UnitsList selectedUnits={selectedUnits} frameStats={frameStats["player_0"]} units={frame.units["player_0"]} factories={frame.factories["player_0"]}/>
            <UnitsList selectedUnits={selectedUnits} frameStats={frameStats["player_1"]} units={frame.units["player_1"]} factories={frame.factories["player_1"]}/>
          </div>
          <TileView viewedTilePos={viewedTilePos} />
          <div className={s.chartWrapper}><Charts /></div>
        </div>
      </div>
      </>
    )
  });