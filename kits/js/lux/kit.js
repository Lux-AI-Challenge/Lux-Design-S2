const fs = require('fs');
const readline = require('readline');

// Create parser and use ',' as the delimiter between commands being sent by the `Match` and `MatchEngine`
// const Parser = require('./parser');
// const {
//   GameMap
// } = require('./map');
// const {
//   INPUT_CONSTANTS
// } = require('./io');
// const {
//   Player,
//   City,
//   Unit,
// } = require('./game_objects');
const GAME_CONSTANTS = require('./game_constants');
// const parse = new Parser(' ');

/**
 * Agent for sequential `Designs`
 */
class Agent {
  _setup() {

    // Prepare to read input
    const rl = readline.createInterface({
      input: process.stdin,
      output: null,
    });

    let buffer = [];
    let currentResolve;
    const makePromise = function () {
      return new Promise((resolve) => {
        currentResolve = resolve;
      });
    };
    // on each line, push line to buffer
    rl.on('line', (line) => {
      buffer.push(line);
      currentResolve();
      currentPromise = makePromise();
    });

    // The current promise for retrieving the next line
    let currentPromise = makePromise();

    // with await, we pause process until there is input
    const getLine = async () => {
      return new Promise(async (resolve) => {
        while (buffer.length === 0) {
          // pause while buffer is empty, continue if new line read
          await currentPromise;
        }
        // once buffer is not empty, resolve the most recent line in stdin, and remove it
        resolve(buffer.shift());
      });
    };
    this.getLine = getLine;
  }

  /**
   * Constructor for a new agent
   */
  constructor() {
    this._setup();
  }

  /**
   * Initialize Agent
   */
  async initialize() {

    // get agent ID
    // const input = (await this.getLine());
    // // console.error({input})
    
    // this.agent = input.player;

    // this.gameState = {
    //   // id,
    //   // map,
    //   // players,
    //   step: input.step,
    // };
  }
  /**
   * Updates agent's own known state of `Match`
   * User should edit this according to their `Design`.
   */
  async update() {
    // this.gameState.turn++;
    // wait for the engine to send any updates
    await this.retrieveUpdates();
  }

  resetPlayerStates() {
    // let players = this.gameState.players;
    // players[0].units = [];
    // players[0].cities = new Map();
    // players[0].cityTileCount = 0;
    // players[1].units = [];
    // players[1].cities = new Map();
    // players[1].cityTileCount = 0;
  }
  async retrieveUpdates() {
    // this.resetPlayerStates();
    const input = JSON.parse(await this.getLine());
    this.last_input = input;
    this.step = parseInt(input["step"]);
    this.agent = input["player"];
    // // TODO: this can be optimized. we only reset because some resources get removed
    // this.gameState.map = new GameMap(this.gameState.map.width, this.gameState.map.height);
    // while (true) {
    //   let update = (await this.getLine());
    //   if (update.str === INPUT_CONSTANTS.DONE) {
    //     break;
    //   }
    //   const inputIdentifier = update.nextStr();
    //   switch (inputIdentifier) {
    //     case INPUT_CONSTANTS.RESEARCH_POINTS: {
    //       const team = update.nextInt();
    //       this.gameState.players[team].researchPoints = update.nextInt();
    //       break;
    //     }
    //     case INPUT_CONSTANTS.RESOURCES: {
    //       const type = update.nextStr();
    //       const x = update.nextInt();
    //       const y = update.nextInt();
    //       const amt = update.nextInt();
    //       this.gameState.map._setResource(type, x, y, amt);
    //       break;
    //     }
    //     case INPUT_CONSTANTS.UNITS: {
    //       const unittype = update.nextInt();
    //       const team = update.nextInt();
    //       const unitid = update.nextStr();
    //       const x = update.nextInt();
    //       const y = update.nextInt();
    //       const cooldown = update.nextFloat();
    //       const wood = update.nextInt();
    //       const coal = update.nextInt();
    //       const uranium = update.nextInt();
    //       this.gameState.players[team].units.push(new Unit(team, unittype, unitid, x, y, cooldown, wood, coal, uranium));
    //       break;
    //     }
    //     case INPUT_CONSTANTS.CITY: {
    //       const team = update.nextInt();
    //       const cityid = update.nextStr();
    //       const fuel = update.nextFloat();
    //       const lightUpkeep = update.nextFloat();
    //       this.gameState.players[team].cities.set(cityid, new City(team, cityid, fuel, lightUpkeep));
    //       break;
    //     }
    //     case INPUT_CONSTANTS.CITY_TILES: {
    //       const team = update.nextInt();
    //       const cityid = update.nextStr();
    //       const x = update.nextInt();
    //       const y = update.nextInt();
    //       const cooldown = update.nextFloat();
    //       const city = this.gameState.players[team].cities.get(cityid);
    //       const citytile = city.addCityTile(x, y, cooldown);
    //       this.gameState.map.getCell(x, y).citytile = citytile;
    //       this.gameState.players[team].cityTileCount += 1;
    //       break;
    //     }
    //     case INPUT_CONSTANTS.ROADS: {
    //       const x = update.nextInt();
    //       const y = update.nextInt();
    //       const road = update.nextFloat();
    //       this.gameState.map.getCell(x, y).road = road;
    //       break;
    //     }
    //   }
    // }
  }
}

const annotate = {
  circle: (x, y) => {
    return `dc ${x} ${y}`
  },
  x: (x, y) => {
    return `dx ${x} ${y}`
  },
  line: (x1, y1, x2, y2) => {
    return `dl ${x1} ${y1} ${x2} ${y2}`
  },
  text: (x1, y1, message, fontsize = 16) => {
    return `dt ${x1} ${y1} '${message}' ${fontsize}`
  },
  sidetext: (message) => {
    return `dst '${message}'`
  }
}

module.exports = {
  Agent,
  annotate
};