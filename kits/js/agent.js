const { processObs } = require("./lux/obs");
const { setup } = require("./lux/setup");
const {myTurnToPlaceFactory} = require("./lux/utils");
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
    // if ice[x][y] > 0, then there is an ice tile at (x, y)
    const ice = obs["board"]["ice"];
    // if ore[x][y] > 0, then there is an ore tile at (x, y)
    const ore = obs["board"]["ore"];

    if (step == 0) {
      // # bid 0 to not waste resources bidding and declare as the default faction
      let faction = "MotherMars";
      if (this.player == "player_1") {
        faction = "AlphaStrike";
      }
      return {
        faction,
        bid: 0,
      };
    } else {
      // decide on where to spawn the next factory. Returning an empty dict() will skip your factory placement

      // how much water and metal you have in your starting pool to give to new factories
      const water_left = obs["teams"][this.player]["water"];
      const metal_left = obs["teams"][this.player]["metal"];
      // how many factories you have left to place
      const factories_to_place =
        obs["teams"][this.player]["factories_to_place"];
      // obs["teams"][this.opp_player] has the same information but for the other team
      // potential spawnable locations in your half of the map
      const possibleSpawns = [];
      const width = obs["board"]["valid_spawns_mask"].length;
      const height = obs["board"]["valid_spawns_mask"][0].length;
      for (let i = 0; i < width; i++) {
        for (let j = 0; j < height; j++) {
          if (obs["board"]["valid_spawns_mask"][i][j] === true) {
            possibleSpawns.push([i, j]);
          }
        }
      }
      // whether it is your turn to place a factory
      const myTurnToPlace = myTurnToPlaceFactory(obs["teams"][this.player].place_first, step)
      if (factories_to_place > 0 && myTurnToPlace) {
        // we will spawn our factory in a random location with 150 metal and water if it is our turn to place
        const spawn_loc =
          possibleSpawns[
            parseInt(Math.floor(Math.random() * possibleSpawns.length))
          ];
        return { spawn: spawn_loc, metal: 150, water: 150 };
      }
      return {};
    }
  }

  act(step) {
    /**
      Logic here to make actions for the rest of the game.
    */
    const obs = this.gameState;
    const actions = {};

    // NOTE that if you set FORWARD_SIM in main.py > 0, then obs will be a list of game states;

    // various maps to help aid in decision making
    const rubble = obs["board"]["rubble"];
    // if ice[x][y] > 0, then there is an ice tile at (x, y)
    const ice = obs["board"]["ice"];
    // if ore[x][y] > 0, then there is an ore tile at (x, y)
    const ore = obs["board"]["ore"];

    // lichen[x][y] = amount of lichen at tile (x, y)
    const lichen = obs["board"]["lichen"];
    // lichenStrains[x][y] = the strain id of the lichen at tile (x, y). Each strain id is
    // associated with a single factory and cannot mix with other strains.
    // factory.strain_id defines the factory's strain id
    const lichenStrains = obs["board"]["lichen_strains"];

    // units and factories for your team and the opposition team
    const units = obs["units"][this.player];
    const opp_units = obs["units"][this.opp_player];
    const factories = obs["factories"][this.player];
    const opp_factories = obs["factories"][this.opp_player];

    // iterate over all active factories
    for (const [unit_id, factory] of Object.entries(factories)) {
      if (step % 4 == 0 && step > 1) {
        actions[unit_id] = Math.floor(Math.random() * 2);
      } else {
        actions[unit_id] = 2;
      }
    }
    for (const [unit_id, unit] of Object.entries(units)) {
      const pos = unit["pos"];
      const moveCost = unit.moveCost(this.gameState, 1);
      if (unit.power > moveCost + unit.actionQueueCost(this.gameState)) {
        actions[unit_id] = [unit.move(Math.floor(Math.random() * 5))];
      }
      
    }
    return actions;
  }

  async update() {
    const input = JSON.parse(await this.getLine());
    this.last_input = input;
    this.step = parseInt(input["step"]);
    this.real_env_steps = parseInt(input["obs"]["real_env_steps"]);
    if (this.step === 0) {
      this.env_cfg = input["info"]["env_cfg"];
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

module.exports = {
  Agent,
};
