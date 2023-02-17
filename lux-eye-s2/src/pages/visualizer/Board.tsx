import { useHover, useMergedRef, useMouse } from '@mantine/hooks';
import { useCallback, useEffect, useState } from 'react';
import { Factory, Robot, RobotType, Step, Tile } from '../../episode/model';
import { useStore } from '../../store';
import { getTeamColor } from '../../utils/colors';

interface SizeConfig {
  gutterSize: number;
  tileSize: number;
  boardSize: number;
  tilesPerSide: number;
}

interface ThemeConfig {
  minimalTheme: boolean;
}

type Config = SizeConfig & ThemeConfig;

function getSizeConfig(maxWidth: number, step: Step): SizeConfig {
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

function tileToCanvas(sizes: SizeConfig, tile: Tile): [number, number] {
  return [
    (tile.x + 1) * sizes.gutterSize + tile.x * sizes.tileSize,
    (tile.y + 1) * sizes.gutterSize + tile.y * sizes.tileSize,
  ];
}

function scale(value: number, relativeMin: number, relativeMax: number): number {
  const clampedValue = Math.max(Math.min(value, relativeMax), relativeMin);
  return (clampedValue - relativeMin) / (relativeMax - relativeMin);
}

function drawTileBackgrounds(ctx: CanvasRenderingContext2D, config: Config, step: Step): void {
  const board = step.board;
  const isDay = step.step < 0 || step.step % 50 < 30;

  const teamStrains = new Map<number, number>();
  for (let i = 0; i < 2; i++) {
    for (const strain of step.teams[i].strains) {
      teamStrains.set(strain, i);
    }
  }

  for (let tileY = 0; tileY < config.tilesPerSide; tileY++) {
    for (let tileX = 0; tileX < config.tilesPerSide; tileX++) {
      const [canvasX, canvasY] = tileToCanvas(config, { x: tileX, y: tileY });

      ctx.fillStyle = 'white';
      ctx.fillRect(canvasX, canvasY, config.tileSize, config.tileSize);

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
      ctx.fillRect(canvasX, canvasY, config.tileSize, config.tileSize);

      const lichen = board.lichen[tileY][tileX];
      if (lichen > 0) {
        const team = teamStrains.get(board.strains[tileY][tileX])!;
        ctx.fillStyle = getTeamColor(team, 0.1 + scale(lichen, 0, 100) * 0.4);
        ctx.fillRect(canvasX, canvasY, config.tileSize, config.tileSize);
      }
    }
  }

  ctx.restore();
}

function drawFactory(
  ctx: CanvasRenderingContext2D,
  config: Config,
  factory: Factory,
  team: number,
  selectedTile: Tile | null,
): void {
  const [canvasX, canvasY] = tileToCanvas(config, {
    x: factory.tile.x - 1,
    y: factory.tile.y - 1,
  });

  const size = config.tileSize * 3 + config.gutterSize * 2;
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
  config: Config,
  robot: Robot,
  team: number,
  selectedTile: Tile | null,
): void {
  const [canvasX, canvasY] = tileToCanvas(config, robot.tile);

  const isSelected = selectedTile !== null && robot.tile.x === selectedTile.x && robot.tile.y === selectedTile.y;

  if (robot.type === RobotType.Light) {
    ctx.fillStyle = getTeamColor(team, 1.0);
    ctx.strokeStyle = 'black';
    ctx.lineWidth = isSelected ? 2 : 1;

    const radius = config.tileSize / 2 - 1;

    ctx.beginPath();
    ctx.arc(canvasX + config.tileSize / 2, canvasY + config.tileSize / 2, radius, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
  } else {
    const borderSize = isSelected ? 1 : 2;

    ctx.fillStyle = 'black';
    ctx.fillRect(canvasX, canvasY, config.tileSize, config.tileSize);

    ctx.fillStyle = getTeamColor(team, 1.0);
    ctx.fillRect(
      canvasX + borderSize,
      canvasY + borderSize,
      config.tileSize - borderSize * 2,
      config.tileSize - borderSize * 2,
    );
  }

  ctx.restore();
}

function drawSelectedTile(ctx: CanvasRenderingContext2D, config: Config, selectedTile: Tile): void {
  const [canvasX, canvasY] = tileToCanvas(config, selectedTile);

  ctx.fillStyle = 'black';

  ctx.fillRect(
    canvasX - config.gutterSize,
    canvasY - config.gutterSize,
    config.tileSize + config.gutterSize * 2,
    config.gutterSize,
  );

  ctx.fillRect(
    canvasX - config.gutterSize,
    canvasY + config.tileSize,
    config.tileSize + config.gutterSize * 2,
    config.gutterSize,
  );

  ctx.fillRect(
    canvasX - config.gutterSize,
    canvasY - config.gutterSize,
    config.gutterSize,
    config.tileSize + config.gutterSize * 2,
  );

  ctx.fillRect(
    canvasX + config.tileSize,
    canvasY - config.gutterSize,
    config.gutterSize,
    config.tileSize + config.gutterSize * 2,
  );

  ctx.restore();
}

function drawBoard(ctx: CanvasRenderingContext2D, config: Config, step: Step, selectedTile: Tile | null): void {
  ctx.save();

  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, config.boardSize, config.boardSize);
  ctx.restore();

  drawTileBackgrounds(ctx, config, step);

  for (let i = 0; i < 2; i++) {
    for (const factory of step.teams[i].factories) {
      drawFactory(ctx, config, factory, i, selectedTile);
    }

    for (const robot of step.teams[i].robots) {
      drawRobot(ctx, config, robot, i, selectedTile);
    }
  }

  if (selectedTile !== null) {
    drawSelectedTile(ctx, config, selectedTile);
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

  const minimalTheme = useStore(state => state.minimalTheme);

  const [sizeConfig, setSizeConfig] = useState<SizeConfig>({
    gutterSize: 0,
    tileSize: 0,
    boardSize: 0,
    tilesPerSide: 0,
  });

  const step = episode!.steps[turn];

  const onMouseLeave = useCallback(() => {
    setSelectedTile(null, true);
  }, []);

  useEffect(() => {
    const newSizeConfig = getSizeConfig(maxWidth, step);
    if (
      newSizeConfig.gutterSize !== sizeConfig.gutterSize ||
      newSizeConfig.tileSize !== sizeConfig.tileSize ||
      newSizeConfig.boardSize !== sizeConfig.boardSize ||
      newSizeConfig.tilesPerSide !== sizeConfig.tilesPerSide
    ) {
      setSizeConfig(newSizeConfig);
    }
  }, [maxWidth, episode]);

  useEffect(() => {
    if (!hovered) {
      return;
    }

    for (let tileY = 0; tileY < sizeConfig.tilesPerSide; tileY++) {
      for (let tileX = 0; tileX < sizeConfig.tilesPerSide; tileX++) {
        const tile = { x: tileX, y: tileY };
        const [canvasX, canvasY] = tileToCanvas(sizeConfig, tile);

        if (
          mouseX >= canvasX &&
          mouseX < canvasX + sizeConfig.tileSize &&
          mouseY >= canvasY &&
          mouseY < canvasY + sizeConfig.tileSize
        ) {
          setSelectedTile(tile, true);
          return;
        }
      }
    }
  }, [sizeConfig, mouseX, mouseY, hovered]);

  useEffect(() => {
    if (sizeConfig.tileSize <= 0) {
      return;
    }

    const ctx = canvasMouseRef.current.getContext('2d')!;

    const config = {
      ...sizeConfig,
      minimalTheme,
    };

    drawBoard(ctx, config, step, selectedTile);
  }, [step, sizeConfig, selectedTile, minimalTheme]);

  return (
    <canvas ref={canvasRef} width={sizeConfig.boardSize} height={sizeConfig.boardSize} onMouseLeave={onMouseLeave} />
  );
}
