import { useStore, useStoreKeys } from "@/store";
import s from "./styles.module.scss";
interface TileViewProps {
  viewedTilePos: {x: number, y: number} | null;
  // setHoveredTilePos: any
  // handleOnMouseEnterTile: any;
}
const mapWidth = 64;
const rows = Array.from({ length: mapWidth });
const cols = Array.from({ length: mapWidth });
export const TileView = ({viewedTilePos}: TileViewProps) => {
  const replay = useStore((state) => state.replay)!; // game map should only get rendered when replay is non-null
    const { turn, speed } = useStoreKeys("turn", "speed");
    const frame = replay.states[turn];
    const frameZero = replay.states[0];
    if (!viewedTilePos) {
      return (
        <><div className={s.tileview}></div></>
      )
    }
    return (
      <>
      <div className={s.tileview}>
        <h3>({viewedTilePos.x}, {viewedTilePos.y})</h3>
      </div>
      </>
    )

}