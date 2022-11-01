// TODO - update this for official visualizer when factions are included

import { Player } from "@/types/replay/player";
import { Faction } from "@/types/replay/team";

export const getColor = (agent: Player, faction: Faction, secondary: boolean = false) => {
  if (agent == "player_0") {
    return "#E04128"
  } else {
    return "#007051"
  }
}