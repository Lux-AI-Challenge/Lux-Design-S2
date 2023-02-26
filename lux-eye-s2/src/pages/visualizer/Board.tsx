import { useHover, useMergedRef, useMouse } from '@mantine/hooks';
import { useCallback, useEffect, useState } from 'react';
import seedrandom from 'seedrandom';
import factoryGreen from '../../assets/factory_green.svg';
import factoryRed from '../../assets/factory_red.svg';
import ice0 from '../../assets/ice0.png';
import ice1 from '../../assets/ice1.png';
import heavy0 from '../../assets/robots/heavy0.png';
import heavy1 from '../../assets/robots/heavy1.png';
import light0 from '../../assets/robots/light0.png';
import light1 from '../../assets/robots/light1.png';
import rubble100 from '../../assets/rubble/r100.png';
import rubble20 from '../../assets/rubble/r20.png';
import rubble40 from '../../assets/rubble/r40.png';
import rubble60 from '../../assets/rubble/r60.png';
import rubble80 from '../../assets/rubble/r80.png';
import { Factory, Robot, RobotType, Step, Tile } from '../../episode/model';
import { useStore } from '../../store';
import { getTeamColor } from '../../utils/colors';
import { lichenTilePaths } from './assets';
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
  lichenTiles: HTMLImageElement[];
  rubbleTiles: HTMLImageElement[];
  iceTiles: HTMLImageElement[];
  oreTiles: HTMLImageElement[];
  lights: HTMLImageElement[];
  heavies: HTMLImageElement[];
}

type Config = SizeConfig & ThemeConfig & AssetConfig;

function getSizeConfig(maxWidth: number, step: Step): SizeConfig {
  const gutterSize = 1;
  const tilesPerSide = step.board.rubble.length;
  // maxWidth =00;
  let tileSize = Math.floor(Math.sqrt(maxWidth));
  let boardSize = tileSize * tilesPerSide + gutterSize * (tilesPerSide + 1);

  while (boardSize > maxWidth) {
    tileSize--;
    boardSize -= tilesPerSide;
  }
  // boardSize = 1400;

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
const randomIds = new Map();
for (let i = 0; i < 48; i++) {
  for (let j = 0; j < 48; j++) {
    randomIds.set(`${i},${j}`, Math.floor(Math.random() * 8));
  }
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
      // ctx.fillStyle = 'white';
      // ctx.fillRect(canvasX, canvasY, config.tileSize, config.tileSize);
      const tilePadding = 2;
      // ctx.filter = 'hue-rotate(-100deg)';
      if (!config.minimalTheme && config.lichenTiles && config.rubbleTiles && config.iceTiles && config.oreTiles) {
        let color: string;
        if (board.ice[tileY][tileX] > 0) {
          let type = 1;
          const rng = seedrandom(`${tileX * tileY}`);
          if (rng() < 0.25) {
            type = 0;
          }
          ctx.drawImage(
            config.iceTiles[type],
            canvasX - tilePadding,
            canvasY - tilePadding,
            config.tileSize + tilePadding * 2,
            config.tileSize + tilePadding * 2,
          );
        } else if (board.ore[tileY][tileX] > 0) {
          color = '#2c3e50';
          ctx.fillStyle = color;
          ctx.fillRect(canvasX, canvasY, config.tileSize, config.tileSize);
        } else if (board.lichen[tileY][tileX] == 0) {
          const bracket = Math.ceil(board.rubble[tileY][tileX] / 20);
          ctx.drawImage(
            config.rubbleTiles[bracket],
            canvasX - tilePadding + 0,
            canvasY - tilePadding + 0,
            config.tileSize + tilePadding * 2,
            config.tileSize + tilePadding * 2,
          );
          // ctx.drawImage(config.lichenTiles[0], canvasX - 2, canvasY - 2, config.tileSize + 4, config.tileSize + 4);

          // const rgb = isDay ? 150 : 75;
          // const base = isDay ? 0.1 : 0.2;
          // color = `rgba(${rgb}, ${rgb}, ${rgb}, ${base + scale(board.rubble[tileY][tileX], 0, 100) * (1 - base)})`;
        }
      } else {
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
      }
      // ctx.filter = 'none';

      const lichen = board.lichen[tileY][tileX];
      // ctx.filter = 'hue-rotate(90deg)';
      if (lichen > 0) {
        const team = teamStrains.get(board.strains[tileY][tileX]);
        if (team !== undefined) {
          if (!config.minimalTheme && config.lichenTiles) {
            const bracket = Math.ceil(lichen / 20);
            const ID = bracket; // * 8 + randomIds.get(`${tileX},${tileY}`);
            // console.log({bracket, ID, }, config.lichenTiles.length)

            ctx.drawImage(
              config.lichenTiles[ID],
              canvasX - tilePadding + 0,
              canvasY - tilePadding + 0,
              config.tileSize + tilePadding * 2,
              config.tileSize + tilePadding * 2,
            );
          } else {
            ctx.fillStyle = getTeamColor(team, 0.1 + scale(lichen, 0, 100) * 0.4);
            ctx.fillRect(canvasX, canvasY, config.tileSize, config.tileSize);
          }
        }
      }
      // ctx.filter = 'none';
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
  if (!config.minimalTheme && config.factories) {
    ctx.drawImage(config.factories[team], canvasX + config.tileSize / 2, canvasY, size - config.tileSize, size);
  } else {
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
  let tilePadding = 3;
  if (robot.type === RobotType.Light) {
    if (!config.minimalTheme) {
      tilePadding = 0;
      ctx.drawImage(
        config.lights[team],
        canvasX - tilePadding + 2,
        canvasY - tilePadding - 1,
        config.tileSize + tilePadding * 2 - 4,
        config.tileSize + tilePadding * 2,
      );
    } else {
      ctx.fillStyle = getTeamColor(team, 1.0);
      ctx.strokeStyle = 'black';
      ctx.lineWidth = isSelected ? 2 : 1;

      const radius = config.tileSize / 2 - 1;

      ctx.beginPath();
      ctx.arc(canvasX + config.tileSize / 2, canvasY + config.tileSize / 2, radius, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
    }
  } else {
    if (!config.minimalTheme) {
      tilePadding = 1;
      ctx.drawImage(
        config.heavies[team],
        canvasX - tilePadding,
        canvasY - tilePadding,
        config.tileSize + tilePadding * 2,
        config.tileSize + tilePadding * 2,
      );
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

  const [assetConfig, setAssetConfig] = useState<AssetConfig>({
    factories: [],
    lights: [],
    heavies: [],
    lichenTiles: [],
    oreTiles: [],
    iceTiles: [],
    rubbleTiles: [],
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
    const factories: HTMLImageElement[] = [];
    const lichenTiles: HTMLImageElement[] = [];
    const rubbleTiles: HTMLImageElement[] = [];
    const heavies: HTMLImageElement[] = [];
    const lights: HTMLImageElement[] = [];
    const iceTiles: HTMLImageElement[] = [];
    const oreTiles: HTMLImageElement[] = [];

    for (const image of [factoryRed, factoryGreen]) {
      const elem = document.createElement('img');
      elem.src = image;
      factories.push(elem);
    }
    for (const image of lichenTilePaths) {
      const elem = document.createElement('img');
      elem.src = image;
      lichenTiles.push(elem);
    }
    for (const image of [lichenTilePaths[0], rubble20, rubble40, rubble60, rubble80, rubble100]) {
      const elem = document.createElement('img');
      elem.src = image;
      rubbleTiles.push(elem);
    }
    for (const image of [light0, light1]) {
      const elem = document.createElement('img');
      elem.src = image;
      lights.push(elem);
    }
    for (const image of [heavy0, heavy1]) {
      const elem = document.createElement('img');
      elem.src = image;
      heavies.push(elem);
    }
    for (const image of [ice0, ice1]) {
      const elem = document.createElement('img');
      elem.src = image;
      iceTiles.push(elem);
    }

    setAssetConfig({ factories, lichenTiles, rubbleTiles, heavies, lights, iceTiles, oreTiles });

    return () => {
      for (const image of factories) {
        image.remove();
      }
      for (const image of lichenTiles) {
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
      ...assetConfig,
    };

    drawBoard(ctx, config, step, selectedTile);
  }, [step, sizeConfig, selectedTile, minimalTheme]);

  return (
    <canvas ref={canvasRef} width={sizeConfig.boardSize} height={sizeConfig.boardSize} onMouseLeave={onMouseLeave} />
  );
}
