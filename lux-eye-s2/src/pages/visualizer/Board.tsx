import { useHover, useMergedRef, useMouse } from '@mantine/hooks';
import { useCallback, useEffect, useState } from 'react';
import { Factory, Robot, RobotType, Step, Tile } from '../../episode/model';
import { useStore } from '../../store';
import { getTeamColor } from '../../utils/colors';

interface Sizes {
  gutterSize: number;
  tileSize: number;
  boardSize: number;
  tilesPerSide: number;
}

function getSizes(maxWidth: number, step: Step): Sizes {
  const gutterSize = 1;
  const tilesPerSide = step.board.rubble.length;

  let tileSize = Math.floor(Math.sqrt(maxWidth));
  let boardSize = tileSize * tilesPerSide + gutterSize * (tilesPerSide + 1);

  while (boardSize > maxWidth) {
    tileSize--;
    boardSize -= tilesPerSide;
  }

  return {
    gutterSize,
    tileSize,
    boardSize,
    tilesPerSide,
  };
}

function tileToCanvas(sizes: Sizes, tile: Tile): [number, number] {
  return [
    (tile.x + 1) * sizes.gutterSize + tile.x * sizes.tileSize,
    (tile.y + 1) * sizes.gutterSize + tile.y * sizes.tileSize,
  ];
}

function scale(value: number, relativeMin: number, relativeMax: number): number {
  const clampedValue = Math.max(Math.min(value, relativeMax), relativeMin);
  return (clampedValue - relativeMin) / (relativeMax - relativeMin);
}

function drawTileBackgrounds(ctx: CanvasRenderingContext2D, sizes: Sizes, step: Step): void {
  const board = step.board;
  const isDay = step.step < 0 || step.step % 50 < 30;

  const teamStrains = new Map<number, number>();
  for (let i = 0; i < 2; i++) {
    for (const factory of step.teams[i].factories) {
      teamStrains.set(factory.strain, i);
    }
  }

  for (let tileY = 0; tileY < sizes.tilesPerSide; tileY++) {
    for (let tileX = 0; tileX < sizes.tilesPerSide; tileX++) {
      const [canvasX, canvasY] = tileToCanvas(sizes, { x: tileX, y: tileY });

      ctx.fillStyle = 'white';
      ctx.fillRect(canvasX, canvasY, sizes.tileSize, sizes.tileSize);

      let color: string;
      if (board.ice[tileY][tileX] > 0) {
        color = '#48dbfb';
      } else if (board.ore[tileY][tileX] > 0) {
        color = '#2c3e50';
      } else {
        const rgb = isDay ? 150 : 75;
        const base = isDay ? 0.1 : 0.2;
        color = `rgba(${rgb}, ${rgb}, ${rgb}, ${base + scale(board.rubble[tileY][tileX], 0, 100) * (1 - base)})`;
      }

      ctx.fillStyle = color;
      ctx.fillRect(canvasX, canvasY, sizes.tileSize, sizes.tileSize);

      const lichen = board.lichen[tileY][tileX];
      if (lichen > 0) {
        const team = teamStrains.get(board.strains[tileY][tileX]);
        if (team !== undefined) {
          ctx.fillStyle = getTeamColor(team, 0.1 + scale(lichen, 0, 100) * 0.4);
          ctx.fillRect(canvasX, canvasY, sizes.tileSize, sizes.tileSize);
        }
      }
    }
  }

  ctx.restore();
}

function drawFactory(
  ctx: CanvasRenderingContext2D,
  sizes: Sizes,
  factory: Factory,
  team: number,
  selectedTile: Tile | null,
): void {
  const [canvasX, canvasY] = tileToCanvas(sizes, {
    x: factory.tile.x - 1,
    y: factory.tile.y - 1,
  });

  const size = sizes.tileSize * 3 + sizes.gutterSize * 2;
  const isSelected =
    selectedTile !== null &&
    Math.abs(factory.tile.x - selectedTile.x) <= 1 &&
    Math.abs(factory.tile.y - selectedTile.y) <= 1;

  const borderSize = 2;

  ctx.fillStyle = 'white';
  ctx.fillRect(canvasX, canvasY, size, size);

  ctx.fillStyle = getTeamColor(team, 0.75);
  ctx.fillRect(canvasX, canvasY, size, size);

  ctx.fillStyle = isSelected ? 'black' : getTeamColor(team, 1.0);
  ctx.fillRect(canvasX, canvasY, size, borderSize);
  ctx.fillRect(canvasX, canvasY, borderSize, size);
  ctx.fillRect(canvasX, canvasY + size - borderSize, size, borderSize);
  ctx.fillRect(canvasX + size - borderSize, canvasY, borderSize, size);

  ctx.restore();
}

function drawRobot(
  ctx: CanvasRenderingContext2D,
  sizes: Sizes,
  robot: Robot,
  team: number,
  selectedTile: Tile | null,
): void {
  const [canvasX, canvasY] = tileToCanvas(sizes, robot.tile);

  const isSelected = selectedTile !== null && robot.tile.x === selectedTile.x && robot.tile.y === selectedTile.y;

  if (robot.type === RobotType.Light) {
    ctx.fillStyle = getTeamColor(team, 1.0);
    ctx.strokeStyle = 'black';
    ctx.lineWidth = isSelected ? 2 : 1;

    const radius = sizes.tileSize / 2 - 1;

    ctx.beginPath();
    ctx.arc(canvasX + sizes.tileSize / 2, canvasY + sizes.tileSize / 2, radius, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
  } else {
    const borderSize = isSelected ? 1 : 2;

    ctx.fillStyle = 'black';
    ctx.fillRect(canvasX, canvasY, sizes.tileSize, sizes.tileSize);

    ctx.fillStyle = getTeamColor(team, 1.0);
    ctx.fillRect(
      canvasX + borderSize,
      canvasY + borderSize,
      sizes.tileSize - borderSize * 2,
      sizes.tileSize - borderSize * 2,
    );
  }

  ctx.restore();
}

function drawSelectedTile(ctx: CanvasRenderingContext2D, sizes: Sizes, selectedTile: Tile): void {
  const [canvasX, canvasY] = tileToCanvas(sizes, selectedTile);

  ctx.fillStyle = 'black';

  ctx.fillRect(
    canvasX - sizes.gutterSize,
    canvasY - sizes.gutterSize,
    sizes.tileSize + sizes.gutterSize * 2,
    sizes.gutterSize,
  );

  ctx.fillRect(
    canvasX - sizes.gutterSize,
    canvasY + sizes.tileSize,
    sizes.tileSize + sizes.gutterSize * 2,
    sizes.gutterSize,
  );

  ctx.fillRect(
    canvasX - sizes.gutterSize,
    canvasY - sizes.gutterSize,
    sizes.gutterSize,
    sizes.tileSize + sizes.gutterSize * 2,
  );

  ctx.fillRect(
    canvasX + sizes.tileSize,
    canvasY - sizes.gutterSize,
    sizes.gutterSize,
    sizes.tileSize + sizes.gutterSize * 2,
  );

  ctx.restore();
}

function drawBoard(ctx: CanvasRenderingContext2D, sizes: Sizes, step: Step, selectedTile: Tile | null): void {
  ctx.save();

  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, sizes.boardSize, sizes.boardSize);
  ctx.restore();

  drawTileBackgrounds(ctx, sizes, step);

  for (let i = 0; i < 2; i++) {
    for (const factory of step.teams[i].factories) {
      drawFactory(ctx, sizes, factory, i, selectedTile);
    }

    for (const robot of step.teams[i].robots) {
      drawRobot(ctx, sizes, robot, i, selectedTile);
    }
  }

  if (selectedTile !== null) {
    drawSelectedTile(ctx, sizes, selectedTile);
  }
}

interface BoardProps {
  maxWidth: number;
}

export function Board({ maxWidth }: BoardProps): JSX.Element {
  const { ref: canvasMouseRef, x: mouseX, y: mouseY } = useMouse<HTMLCanvasElement>();
  const { ref: canvasHoverRef, hovered } = useHover<HTMLCanvasElement>();
  const canvasRef = useMergedRef(canvasMouseRef, canvasHoverRef);

  const episode = useStore(state => state.episode);
  const turn = useStore(state => state.turn);

  const selectedTile = useStore(state => state.selectedTile);
  const setSelectedTile = useStore(state => state.setSelectedTile);

  const [sizes, setSizes] = useState<Sizes>({
    gutterSize: 0,
    tileSize: 0,
    boardSize: 0,
    tilesPerSide: 0,
  });

  const step = episode!.steps[turn];

  const onMouseLeave = useCallback(() => {
    setSelectedTile(null);
  }, []);

  useEffect(() => {
    const newSizes = getSizes(maxWidth, step);
    if (
      newSizes.gutterSize !== sizes.gutterSize ||
      newSizes.tileSize !== sizes.tileSize ||
      newSizes.boardSize !== sizes.boardSize ||
      newSizes.tilesPerSide !== sizes.tilesPerSide
    ) {
      setSizes(newSizes);
    }
  }, [maxWidth, episode]);

  useEffect(() => {
    if (!hovered) {
      return;
    }

    for (let tileY = 0; tileY < sizes.tilesPerSide; tileY++) {
      for (let tileX = 0; tileX < sizes.tilesPerSide; tileX++) {
        const tile = { x: tileX, y: tileY };
        const [canvasX, canvasY] = tileToCanvas(sizes, tile);

        if (
          mouseX >= canvasX &&
          mouseX < canvasX + sizes.tileSize &&
          mouseY >= canvasY &&
          mouseY < canvasY + sizes.tileSize
        ) {
          setSelectedTile(tile);
          return;
        }
      }
    }
  }, [sizes, mouseX, mouseY, hovered]);

  useEffect(() => {
    if (sizes.tileSize <= 0) {
      return;
    }

    const ctx = canvasMouseRef.current.getContext('2d')!;
    drawBoard(ctx, sizes, step, selectedTile);
  }, [step, sizes, selectedTile]);

  return <canvas ref={canvasRef} width={sizes.boardSize} height={sizes.boardSize} onMouseLeave={onMouseLeave} />;
}
