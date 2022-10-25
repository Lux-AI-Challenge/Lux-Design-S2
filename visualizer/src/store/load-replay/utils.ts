import { WEATHER_ID_TO_NAME } from "@/constants";
import { FrameStats, KaggleReplay, Replay, ReplayStats } from "@/types/replay";
import { ResourceTile } from "@/types/replay/resource-map";
export function estimateGoodTileWidth(): number {
  let bound = window.innerHeight;
  // if (window.innerWidth * 0.7 < bound) { 
  //   bound = window.innerWidth * 0.7;
  // }
  let tileWidth = Math.floor((bound - 200) / 48) - 2;
  let approxMapWidth = (tileWidth+3) * 48;
  if (approxMapWidth / window.innerWidth > 0.65) {
    // tileWidth -= 1;
    tileWidth = Math.floor(window.innerWidth * 0.65 / 48) - 3;
  }
  approxMapWidth = (tileWidth+3) * 48;
  
  console.log(`Estimated tile width: ${tileWidth}. ${approxMapWidth / window.innerWidth}`)
  return tileWidth;
}
export function convertFromKaggle(kaggleReplay: KaggleReplay): Replay {
  const replay: Replay = {
    meta: {
      teams: [{name: 'Player 0'}, {name: 'Player 1'}],
      weather_events: [],
      real_start: 0,
    },
    observations: [],
    actions: [],
  }
  console.log({kaggleReplay});
  if (kaggleReplay.info.TeamNames) {
    replay.meta.teams[0].name = kaggleReplay.info.TeamNames[0];
    replay.meta.teams[1].name = kaggleReplay.info.TeamNames[1];
  }
  for (let i = 0; i < kaggleReplay.steps.length; i++) {
    const kframe = kaggleReplay.steps[i];
    const obs_str = kframe[0].observation.obs;
    const obs = JSON.parse(obs_str);
    replay.observations.push(obs);
    if (i > 0) {
      replay.actions.push({
        "player_0": kframe[0].action,
        "player_1": kframe[1].action
      })
    }
  }
  return replay;
}
function isKaggleReplay(replay: Replay | KaggleReplay): replay is KaggleReplay {
  return (replay as KaggleReplay).steps !== undefined;
}
export function loadFromObject(replay: Replay | KaggleReplay): Replay {
  const stime = (new Date()).getTime();  
  // TODO: validate that the replay is in the right format (?)
  // re-generate all board frames as necessary
  const hashToPos = (hash: string) => {
    const info = hash.split(",");
    return { x: parseInt(info[0]), y: parseInt(info[1]) };
  };
  const loadedReplay: Replay = {
    meta: {
      teams: [{name: 'Player 0'}, {name: 'Player 1'}],
      weather_events: [],
      real_start: 0,
    },
    observations: [],
    actions: [],
  }
  if (isKaggleReplay(replay)) {
    replay = convertFromKaggle(replay);
    loadedReplay.meta = replay.meta;
  }
  loadedReplay.meta.real_start = -replay.observations[1].real_env_steps + 1;
  const firstBoard = replay.observations[0].board;
  loadedReplay.observations[0] = replay.observations[0];
  loadedReplay.actions = replay.actions;
  for (let i = 1; i < replay.observations.length; i++) {
    const delta_board = replay.observations[i].board;
    // TODO: optimize the deep copy.
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
    loadedReplay.observations[i] = replay.observations[i];
  }
  const etime = (new Date()).getTime();

  let prev_weather = -2;
  let cur_weather = -1;
   // copied from config.py
  loadedReplay.observations[0].weather_schedule.forEach((v, idx) => {
    if (v !== 0) {
      if (cur_weather != prev_weather) {
        // if new weather encountered, add new weather event
        prev_weather = cur_weather
        cur_weather = v;
        loadedReplay.meta.weather_events.push({start: idx, end: idx, name: WEATHER_ID_TO_NAME[v]});
      }
      else {
        loadedReplay.meta.weather_events[loadedReplay.meta.weather_events.length - 1].end = idx;
      }
    } else {
      prev_weather = -2;
      cur_weather = -1;
    }
  });


  console.log(`Loading replay + regeneration took ${(etime - stime)}ms`)
  return loadedReplay;
}

export function computeStatistics(replay: Replay): ReplayStats {
  const stime = (new Date()).getTime();
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
        factoryLichen: {},
        factoryLichenTiles: {},
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
        frameStats[player].factoryLichen[factory.unit_id] = 0;
        frameStats[player].factoryLichenTiles[factory.unit_id] = 0;
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
            frameStats[playerId].factoryLichen[`factory_${strain_id}`] += frame.board.lichen[y][x];
            frameStats[playerId].factoryLichenTiles[`factory_${strain_id}`] += 1;
            break;
          }
        }
        
      }
    }
    stats.frameStats.push(frameStats);
  });
  const etime = (new Date()).getTime();
  console.log(`Precomputation of per-step stats took ${(etime - stime)}ms`)
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

