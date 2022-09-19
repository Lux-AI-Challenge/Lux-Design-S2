import { useStore, useStoreKeys } from "@/store";
import { Player } from "@/types/replay/player";
import { Unit } from "@/types/replay/unit";
import s from "./styles.module.scss";
interface TileViewProps {
  viewedTilePos: { x: number; y: number } | null;
  // setHoveredTilePos: any
  // handleOnMouseEnterTile: any;
}
const mapWidth = 48;
const rows = Array.from({ length: mapWidth });
const cols = Array.from({ length: mapWidth });
export const TileView = ({ viewedTilePos }: TileViewProps) => {
  function toTitleCase(str: string) {
    return str.replace(/\w\S*/g, function (txt) {
      return txt.charAt(0).toUpperCase() + txt.substring(1).toLowerCase();
    });
  }
  const replay = useStore((state) => state.replay)!; // game map should only get rendered when replay is non-null
  const { turn, speed, gameInfo } = useStoreKeys("turn", "speed", "gameInfo");
  const frame = replay.observations[turn];
  const frameZero = replay.observations[0];
  if (!viewedTilePos) {
    return (
      <>
        <div className={s.tileview}></div>
      </>
    );
  }
  // let viewedUnit = gameInfo.posToUnit.get(
  //   `${viewedTilePos.x},${viewedTilePos.y}`
  // );
  return (
    <>
      <div className={s.tileview}>
        <h2>Tile ({viewedTilePos.x}, {viewedTilePos.y})</h2>
        <div>
          <p>Rubble: {frame.board.rubble[viewedTilePos.y][viewedTilePos.x]}</p>
          <p>Lichen: {frame.board.lichen[viewedTilePos.y][viewedTilePos.x]}</p>
          <p>
            Lichen Strain:{" "}
            {frame.board.lichen_strains[viewedTilePos.y][viewedTilePos.x]}
          </p>
        </div>
        {/* {viewedUnit && (
          <div>
            <h4>
              {toTitleCase(viewedUnit.unit_type)} unit: {viewedUnit.unit_id}
            </h4>
            <p>Power: {viewedUnit.power}</p>
            <p>
              Cargo: Ice: {viewedUnit.cargo.ice}, Ore: {viewedUnit.cargo.ore},
              Water: {viewedUnit.cargo.water}, Metal: {viewedUnit.cargo.metal}
            </p>
            <p>Team: {viewedUnit.team_id}</p>
          </div>
        )} */}
      </div>
    </>
  );
};
