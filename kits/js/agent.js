
const { processObs } = require("./lux/obs");
const { setup } = require("./lux/setup");

/**
 * Agent for sequential `Designs`
 */
class Agent {
  constructor() {
    this.getLine = setup();
  }
  async initialize() {
    this.gameState = {};
  }

  earlySetup(step) {
    /**
      Logic here to make actions in the early game. Select faction, bid for an extra factory, and place factories
    */
    const obs = this.gameState;
    // various maps to help aid in decision making over factory placement
    
    const rubble = obs["board"]["rubble"];
    // if ice[y][x] > 0, then there is an ice tile at (x, y)
    const ice = obs["board"]["ice"];
    // if ore[y][x] > 0, then there is an ore tile at (x, y)
    const ore = obs["board"]["ore"];

    if (step == 0) {
      // decide on a faction, and make a bid for the extra factory.
      // Each unit of bid removes one unit of water and metal from your initial pool
      let faction = "MotherMars";
      if (this.player == "player_1") {
        faction = "AlphaStrike";
      }
      console.error({faction})
      return {
        faction,
        bid: 10,
      };
    } else {
      // decide on where to spawn the next factory. Returning an empty dict() will skip your factory placement

      const factories = obs["factories"][this.player];
      const factoryMap = {}
      // iterate over all active factories
      for (const [unit_id, factory] of Object.entries(factories)) {
        factoryMap[`${factory.pos[0]}-${factory.pos[1]}`] = true;
        factoryMap[`${factory.pos[0]-1}-${factory.pos[1]}`] = true;
        factoryMap[`${factory.pos[0]+1}-${factory.pos[1]}`] = true;
        factoryMap[`${factory.pos[0]}-${factory.pos[1]+1}`] = true;
        factoryMap[`${factory.pos[0]}-${factory.pos[1]-1}`] = true;
        factoryMap[`${factory.pos[0]-1}-${factory.pos[1]-1}`] = true;
        factoryMap[`${factory.pos[0]-1}-${factory.pos[1]+1}`] = true;
        factoryMap[`${factory.pos[0]+1}-${factory.pos[1]+1}`] = true;
        factoryMap[`${factory.pos[0]+1}-${factory.pos[1]-1}`] = true;
      }
      // how much water and metal you have in your starting pool to give to new factories
      const water_left = obs["teams"][this.player]["water"];
      const metal_left = obs["teams"][this.player]["metal"];
      // how many factories you have left to place
      const factories_to_place = obs["teams"][this.player]["factories_to_place"];
      // obs["teams"][this.opp_player] has the same information but for the other team
      // potential spawnable locations in your half of the map
      const potential_spawns = obs["board"]["spawns"][this.player];

      // as a naive approach we randomly select a spawn location and spawn a factory there
      let spawn_loc = potential_spawns[parseInt(Math.floor(Math.random() * potential_spawns.length))];
      let tries = 0;
      let [x, y] = spawn_loc;
      while(tries < 10 && factoryMap[`${x}-${y}`]) {
        spawn_loc = potential_spawns[parseInt(Math.floor(Math.random() * potential_spawns.length))];
        [x, y] = spawn_loc;
        tries++;
      }
      return { spawn: spawn_loc, metal: 100, water: 100 };
    }
  }

  act(step) {
    /**
      Logic here to make actions for the rest of the game.
    */
    const obs = this.gameState;
    const actions = {};

    // various maps to help aid in decision making
    const rubble = obs["board"]["rubble"];
    // if ice[y][x] > 0, then there is an ice tile at (x, y)
    const ice = obs["board"]["ice"];
    // if ore[y][x] > 0, then there is an ore tile at (x, y)
    const ore = obs["board"]["ore"];

    // lichen[y][x] = amount of lichen at tile (x, y)
    const lichen = obs["board"]["lichen"];
    // lichenStrains[y][x] = the strain id of the lichen at tile (x, y). Each strain id is
    // associated with a single factory and cannot mix with other strains.
    // factory.strain_id defines the factory's strain id
    const lichenStrains = obs["board"]["lichen_strains"];

    // units and factories for your team and the opposition team
    const units = obs["units"][this.player];
    const opp_units = obs["units"][this.opp_player];
    const factories = obs["factories"][this.player];
    const opp_factories = obs["factories"][this.opp_player];

    // pre compute useful information
    const icePositions = [];
    for (let y = 0; y < ice.length; y++) {
      for (let x = 0; x < ice[y].length; x++) {
        const iceExist = ice[y][x];
        if(iceExist) {
          icePositions.push([x, y]);
        }
      }
    }

    const factoryPositions = [];
    const factoryPositionMap = {};
    for (const [unit_id, factory] of Object.entries(factories)) {
      factoryPositions.push(factory.pos);
      factoryPositionMap[`${factory.pos[0]}-${factory.pos[1]}`] = factory;
      factoryPositionMap[`${factory.pos[0]}-${factory.pos[1]}`] = factory;
      factoryPositionMap[`${factory.pos[0]-1}-${factory.pos[1]}`] = factory;
      factoryPositionMap[`${factory.pos[0]+1}-${factory.pos[1]}`] = factory;
      factoryPositionMap[`${factory.pos[0]}-${factory.pos[1]+1}`] = factory;
      factoryPositionMap[`${factory.pos[0]}-${factory.pos[1]-1}`] = factory;
      factoryPositionMap[`${factory.pos[0]-1}-${factory.pos[1]-1}`] = factory;
      factoryPositionMap[`${factory.pos[0]-1}-${factory.pos[1]+1}`] = factory;
      factoryPositionMap[`${factory.pos[0]+1}-${factory.pos[1]+1}`] = factory;
      factoryPositionMap[`${factory.pos[0]+1}-${factory.pos[1]-1}`] = factory;
    }

    // iterate over all active factories
    for (const [unit_id, factory] of Object.entries(factories)) {
      if(this.player === 'player_0') {
        // console.error(this.player, step, factory.cargo, factory.power, factory.canBuildHeavy(obs));
      }
      if (step % 10 == 0 && step > 1 && factory.cargo['water'] > 500) {
        actions[unit_id] = 2;
      } else if (factory.canBuildHeavy(obs)) {
        actions[unit_id] = factory.buildHeavy();
      }
    }

    const chargeToAmount = 1000;
    const mineIceAmount = 40;

    for (const [unit_id, unit] of Object.entries(units)) {
      const [unitX, unitY] = unit.pos;
      const onIce = ice[unitY][unitX] ? true : false;
      const onFactory = factoryPositionMap[`${unitX}-${unitY}`] ? true : false;
      if(this.player === 'player_0') {
        console.error(this.player, step, unit.pos, 'power', unit.power, 'ice', unit.cargo.ice, 'onice', onIce ? 'Y':'N', 'onFactory', onFactory ? 'Y':'N', unit.actionQueue.length);
      }
      if(unit.actionQueue.length === 0){
        if(onFactory) {
          const factroy = factoryPositionMap[`${unitX}-${unitY}`];
          if(unit.cargo.ice >= mineIceAmount) {
            // transfer ice if has more than mineIceAmount
            actions[unit_id] = [unit.transfer(0, 0, 10, false)];
          } else if(unit.power < chargeToAmount) {
            // charge if no power
            const powerNeeded = chargeToAmount - unit.power;
            if(factroy.power >= powerNeeded) {
              actions[unit_id] = [unit.pickup(4, powerNeeded, false)];
            } else if(factroy.power >= 50) {
              actions[unit_id] = [unit.pickup(4, factroy.power, false)];
            } else {
              actions[unit_id] = [unit.recharge(chargeToAmount, false)];
            }
          } else {
            // else go mine
            const closestIce = getClosestTo(unit.pos, icePositions);
            const direction = getDirectionTo(unit.pos, closestIce);
            actions[unit_id] = [unit.move(direction, false)];
          }
        } else if(onIce) {
          if(unit.cargo.ice >= mineIceAmount) {
            // go back to factory
            const closestFactory = getClosestTo(unit.pos, factoryPositions);
            const direction = getDirectionTo(unit.pos, closestFactory);
            actions[unit_id] = [unit.move(direction, false)];
          } else {
            actions[unit_id] = [unit.dig(false)];
          }
        } else {
          if(unit.cargo.ice >= mineIceAmount) {
            // go back to factory
            const closestFactory = getClosestTo(unit.pos, factoryPositions);
            const direction = getDirectionTo(unit.pos, closestFactory);
            actions[unit_id] = [unit.move(direction, false)];
          } else {
            // go mine
            const closestIce = getClosestTo(unit.pos, icePositions);
            const direction = getDirectionTo(unit.pos, closestIce);
            actions[unit_id] = [unit.move(direction, false)];
          }
        }
      }
    }
    return actions;
  }

  async update() {
    const input = JSON.parse(await this.getLine());
    this.last_input = input;
    this.step = parseInt(input["step"]);
    this.real_env_steps = parseInt(input["obs"]["real_env_steps"])
    if (this.step === 0) {
      this.env_cfg = input["info"]["env_cfg"]
    }
    this.player = input["player"];
    if (this.player == "player_0") {
      this.opp_player = "player_1";
    } else {
      this.opp_player = "player_0";
    }
    this.gameState = processObs(this, this.last_input, this.step);
  }
}

function getRandomInt(min, max) {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function getOppositeDir(direction) {
  if(direction === 0) return 0;
  if(direction === 1) return 3;
  if(direction === 2) return 4;
  if(direction === 3) return 1;
  if(direction === 4) return 2;
}

// direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
function getDirectionTo(source, dist) {
  const [sourceX, sourceY] = source;
  const [distX, distY] = dist;
  if(distX < sourceX) {
    return 4
  } else if(distX > sourceX) {
    return 2
  } else {
    if(distY < sourceY) {
      return 1
    } else if(distY > sourceY) {
      return 3
    }
    return 0
  }
}

function getClosestTo(pos, candidates) {
  let min = Infinity;
  let result = candidates[0];
  const [x, y] = pos;
  for (let i = 0; i < candidates.length; i++) {
    const [cx, cy] = candidates[i];
    // Manhattan distance for grid
    const dist = Math.abs(x - cx) + Math.abs(y - cy);
    if(dist < min) {
      min = dist;
      result = candidates[i];
    }
  }
  return result;
}

module.exports = {
  Agent,
};
