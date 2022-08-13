import { Replay } from "@/types/replay"

export function loadFromObject (replay: Replay): Replay {
  // TODO: validate that the replay is in the right format (?)
  return replay
}

export function loadFromString (replay: string): Replay {
  const replayObject = JSON.parse(replay) as Replay
  return loadFromObject(replayObject)
}

export async function loadFromFile (replay: File): Promise<Replay> {
  // TODO: check to see if file is zipped?
  const contents = await replay.text()
  console.log('read text from file')
  return loadFromString(contents)
}

// not implemented yet. exists for reference to do in the future (maybe)
export function loadFromKaggle () {

}