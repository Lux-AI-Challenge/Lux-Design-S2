import { Controls } from "@/components/Controls";
import { GameMap } from "@/components/GameMap";
import { MouseEventHandler, useState } from "react";

export function ReplayViewer () {
  const [hoveredTilePos, setHoveredTilePos] = useState({x: 0, y: 0});
  const handleOnMouseEnterTile: MouseEventHandler<HTMLDivElement> = (e) => {
    const x = e.currentTarget.getAttribute("x");
    const y = e.currentTarget.getAttribute("y");
    //@ts-ignore
    setHoveredTilePos({ x, y });
  };
  return (
    <>
    <div>
        Hovered: {hoveredTilePos.x}, {hoveredTilePos.y}
      </div>
      <GameMap handleOnMouseEnterTile={handleOnMouseEnterTile}  />
      <Controls />
    </>
  )
}