import { initialProgress, initialReplay } from "./load-replay/constants";
import { initialSelectedEntity } from "./selected-entity/constants";
import { initialAutoplay, initialSpeed } from "./autoplay/constants";
import { initialTurn } from "./turn/constants";

export const initial = {
  replay: initialReplay,
  progress: initialProgress,
 
  turn: initialTurn,

  autoplay: initialAutoplay,
  speed: initialSpeed,
  
  selectedEntity: initialSelectedEntity,

  tileWidth: 12
}