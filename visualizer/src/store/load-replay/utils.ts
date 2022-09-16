import { Replay } from "@/types/replay";
import { ResourceTile } from "@/types/replay/resource-map";

export function loadFromObject(replay: Replay): Replay {
  // TODO: validate that the replay is in the right format (?)
  // re-generate all board frames as necessary
  console.log(JSON.parse(JSON.stringify(replay)));
  const hashToPos = (hash: string) => {
    const info = hash.split(",");
    return { x: parseInt(info[0]), y: parseInt(info[1]) };
  };
  const firstBoard = replay.observations[0].board;
  for (let i = 1; i < replay.observations.length; i++) {
    const delta_board = replay.observations[i].board;
    const board_i = JSON.parse(JSON.stringify(firstBoard));
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
