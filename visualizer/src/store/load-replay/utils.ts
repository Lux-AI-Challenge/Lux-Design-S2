import { FrameStats, Replay, ReplayStats } from "@/types/replay";
import { ResourceTile } from "@/types/replay/resource-map";

export function loadFromObject(replay: Replay): Replay {
  // TODO: validate that the replay is in the right format (?)
  // re-generate all board frames as necessary
  const hashToPos = (hash: string) => {
    const info = hash.split(",");
    return { x: parseInt(info[0]), y: parseInt(info[1]) };
  };
  const firstBoard = replay.observations[0].board;
  for (let i = 1; i < replay.observations.length; i++) {
    const delta_board = replay.observations[i].board;
    // console.log(Object.keys(delta_board))
    const board_i = JSON.parse(JSON.stringify(replay.observations[i - 1].board));
    const delta_keys: ResourceTile[] = ["rubble", "lichen", "lichen_strains"];
    delta_keys.forEach((k: ResourceTile) => {
      for (const hash of Object.keys(delta_board[k])) {
        const { x, y } = hashToPos(hash);
        //@ts-ignore
        board_i[k][y][x] = delta_board[k][hash];
      }
    });
    replay.observations[i].board = board_i;
  }
  console.log(JSON.parse(JSON.stringify(replay)));
  return replay;
}

export function computeStatistics(replay: Replay): ReplayStats {
  console.log("COMPUTE")
  const stats: ReplayStats = {
    frameStats: [],
    mapStats: {
      // TODO
      iceTiles: 0,
      oreTiles: 0
    }
  }
  const players = ["player_0", "player_1"]
  
  replay.observations.forEach((frame) => {
    const playerToLichenStrains: Record<string, Set<number>> = {};
    const frameStats: FrameStats = {
      
    }
    players.forEach((player) => {
      const units = frame.units[player];
      frameStats[player] = {
        units: {heavy: 0, light: 0},
        lichen: 0,
        ore: 0,
        metal: 0,
        water: 0,
        ice: 0,
        storedPower: 0,
      }
      Object.entries(units).forEach(([_, unit]) => {
        if (unit.unit_type == "LIGHT") {
          frameStats[player].units.light += 1;
        }
        else if (unit.unit_type == "HEAVY") {
          frameStats[player].units.heavy += 1;
        }
        Object.entries(unit.cargo).forEach(([resourcetype, x]) => {
          // @ts-ignore
          frameStats[player][resourcetype] += x
        });
        frameStats[player].storedPower += unit.power;
      });
      const ownedLichenStrains: Set<number> = new Set();
      Object.entries(frame.factories[player]).forEach(([_, factory]) => {
        Object.entries(factory.cargo).forEach(([resourcetype, x]) => {
          // @ts-ignore
          frameStats[player][resourcetype] += x
        });
        frameStats[player].storedPower += factory.power;
        ownedLichenStrains.add(factory.strain_id);
      });
      playerToLichenStrains[player] = ownedLichenStrains
    });
    for (let y = 0; y < frame.board.lichen.length; y ++) {
      for (let x = 0; x < frame.board.lichen[0].length; x ++) {
        const strain_id = frame.board.lichen_strains[y][x];
        if (strain_id == -1) continue;
        for (const [playerId, strains] of Object.entries(playerToLichenStrains)) {
          if (strains.has(strain_id)) {
            frameStats[playerId].lichen += frame.board.lichen[y][x];
            break;
          }
        }
        
      }
    }
    stats.frameStats.push(frameStats);
  });
  return stats as ReplayStats;
}


export function loadFromString(replay: string): Replay {
  const replayObject = JSON.parse(replay) as Replay;
  return loadFromObject(replayObject);
}

export async function loadFromFile(replay: File): Promise<Replay> {
  // TODO: check to see if file is zipped?
  const contents = await replay.text();
  console.log("read text from file");
  return loadFromString(contents);
}

// not implemented yet. exists for reference to do in the future (maybe)
export function loadFromKaggle() {}
