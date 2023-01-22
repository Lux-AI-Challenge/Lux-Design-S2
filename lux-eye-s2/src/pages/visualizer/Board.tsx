import { useHover, useMergedRef, useMouse } from '@mantine/hooks';
import { useCallback, useEffect, useState } from 'react';
import factoryGreen from '../../assets/factory-green.svg';
import factoryRed from '../../assets/factory-red.svg';
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

interface AssetConfig {
  factories: HTMLImageElement[];
}

type Config = SizeConfig & ThemeConfig & AssetConfig;

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

  if (!config.minimalTheme && !isDay) {
    ctx.fillStyle = 'rgba(0, 0, 0, 0.25)';
    ctx.fillRect(0, 0, config.boardSize, config.boardSize);
  }

  const teamStrains = new Map<number, number>();
  for (let i = 0; i < 2; i++) {
    for (const factory of step.teams[i].factories) {
      teamStrains.set(factory.strain, i);
    }
  }

  for (let tileY = 0; tileY < config.tilesPerSide; tileY++) {
    for (let tileX = 0; tileX < config.tilesPerSide; tileX++) {
      const [canvasX, canvasY] = tileToCanvas(config, { x: tileX, y: tileY });

      if (config.minimalTheme) {
        ctx.fillStyle = 'white';
        ctx.fillRect(canvasX, canvasY, config.tileSize, config.tileSize);
      }

      let color: string;
      if (board.ice[tileY][tileX] > 0) {
        color = config.minimalTheme ? '#48dbfb' : '#2c9ed3';
      } else if (board.ore[tileY][tileX] > 0) {
        color = config.minimalTheme ? '#2c3e50' : '#daa730';
      } else if (config.minimalTheme) {
        const rgb = isDay ? 150 : 75;
        const base = isDay ? 0.1 : 0.2;
        color = `rgba(${rgb}, ${rgb}, ${rgb}, ${base + scale(board.rubble[tileY][tileX], 0, 100) * (1 - base)})`;
      } else {
        const rubble = board.rubble[tileY][tileX];

        if (rubble === 0) {
          color = 'rgba(255, 255, 255, 0.2)';
        } else {
          color = `rgba(96, 32, 9, ${0.2 + Math.min(rubble / 100, 1) * 0.8})`;
        }
      }

      ctx.fillStyle = color;
      ctx.fillRect(canvasX, canvasY, config.tileSize, config.tileSize);

      const lichen = board.lichen[tileY][tileX];
      if (lichen > 0) {
        const team = teamStrains.get(board.strains[tileY][tileX]);
        if (team !== undefined) {
          if (config.minimalTheme) {
            ctx.fillStyle = getTeamColor(team, 0.1 + scale(lichen, 0, 100) * 0.4, config.minimalTheme);
          } else {
            ctx.fillStyle = `rgba(127, 206, 152, ${lichen / 100})`;
          }

          ctx.fillRect(canvasX, canvasY, config.tileSize, config.tileSize);
        }
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

  if (config.minimalTheme) {
    ctx.fillStyle = 'white';
    ctx.fillRect(canvasX, canvasY, size, size);

    ctx.fillStyle = getTeamColor(team, 0.75, config.minimalTheme);
    ctx.fillRect(canvasX, canvasY, size, size);

    const isSelected =
      selectedTile !== null &&
      Math.abs(factory.tile.x - selectedTile.x) <= 1 &&
      Math.abs(factory.tile.y - selectedTile.y) <= 1;

    const borderSize = 2;

    ctx.fillStyle = isSelected ? 'black' : getTeamColor(team, 1.0, config.minimalTheme);
    ctx.fillRect(canvasX, canvasY, size, borderSize);
    ctx.fillRect(canvasX, canvasY, borderSize, size);
    ctx.fillRect(canvasX, canvasY + size - borderSize, size, borderSize);
    ctx.fillRect(canvasX + size - borderSize, canvasY, borderSize, size);
  } else if (config.factories[team] !== undefined) {
    ctx.drawImage(config.factories[team], canvasX + config.tileSize / 2, canvasY, size - config.tileSize, size);
  }

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
    ctx.fillStyle = getTeamColor(team, 1.0, config.minimalTheme);
    ctx.strokeStyle = config.minimalTheme ? 'black' : 'white';
    ctx.lineWidth = isSelected ? 2 : 1;

    const radius = config.tileSize / 2 - 1;

    ctx.beginPath();
    ctx.arc(canvasX + config.tileSize / 2, canvasY + config.tileSize / 2, radius, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
  } else {
    let borderSize: number;
    if (config.minimalTheme) {
      borderSize = isSelected ? 1 : 2;
      ctx.fillStyle = 'black';
    } else {
      borderSize = 1;
      ctx.fillStyle = 'white';
    }

    ctx.fillRect(canvasX, canvasY, config.tileSize, config.tileSize);

    ctx.fillStyle = getTeamColor(team, 1.0, config.minimalTheme);
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

  ctx.fillStyle = config.minimalTheme ? 'black' : 'white';

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

  ctx.fillStyle = config.minimalTheme ? 'white' : '#ef784f';
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

  const [assetConfig, setAssetConfig] = useState<AssetConfig>({
    factories: [],
  });

  const step = episode!.steps[turn];

  const onMouseLeave = useCallback(() => {
    setSelectedTile(null);
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
    const factories: HTMLImageElement[] = [];

    for (const image of [factoryRed, factoryGreen]) {
      const elem = document.createElement('img');
      elem.src = image;
      factories.push(elem);
    }

    setAssetConfig({ factories });

    return () => {
      for (const image of factories) {
        image.remove();
      }
    };
  }, []);

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
          setSelectedTile(tile);
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
      ...assetConfig,
    };

    drawBoard(ctx, config, step, selectedTile);
  }, [step, sizeConfig, assetConfig, selectedTile, minimalTheme]);

  return (
    <canvas ref={canvasRef} width={sizeConfig.boardSize} height={sizeConfig.boardSize} onMouseLeave={onMouseLeave} />
  );
}
