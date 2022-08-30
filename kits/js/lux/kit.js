const fs = require('fs');
const readline = require('readline');
const {processObs} = require("./obs");
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
    this.gameState = {}
  }
  /**
   * Updates agent's own known state of `Match`
   */
  async update() {
    // wait for the engine to send any updates
    await this.retrieveUpdates();
  }

  resetPlayerStates() {

  }
  async retrieveUpdates() {
    // this.resetPlayerStates();
    const input = JSON.parse(await this.getLine());
    this.last_input = input;
    this.step = parseInt(input["step"]);
    this.player = input["player"];

    this.gameState = processObs(this, this.last_input, this.step);

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