import { TileView } from "@/components/TileView/TileView";
import { useStoreKeys } from "@/store";
import { Player } from "@/types/replay/player";
import React from "react";
import s from "./styles.module.scss";
type InfoPanelProps = {
  viewedTilePos: { x: number; y: number } | null;
}

// move out later
const colors = {
  
}

export const InfoPanel = React.memo(
  ({
    viewedTilePos
  }: InfoPanelProps) => {
    const { gameInfo } = useStoreKeys("gameInfo");
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
              <div>{gameInfo.factoryCounts["player_0"]}</div>
              <div>{gameInfo.unitCounts["player_0"]}</div>
            </div>
            <div><p>Factories</p><p>Robots</p></div>
            <div className={s.value}>
              <div>{gameInfo.factoryCounts["player_1"]}</div>
              <div>{gameInfo.unitCounts["player_1"]}</div>
            </div>
          </div>
          <TileView viewedTilePos={viewedTilePos} />
        </div>
      </div>
      </>
    )
  });