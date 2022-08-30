const kit = require('./lux/kit');
// const GAME_CONSTANTS = require('./lux/game_constants');
// const DIRECTIONS = GAME_CONSTANTS.DIRECTIONS;
// create a new agent
const agent = new kit.Agent();
// const annotate = kit.annotate;

// first initialize the agent, and then proceed to go in a loop waiting for updates and running the AI
agent.initialize().then(async () => {
  while (true) {
    /** Do not edit! **/
    // wait for updates
    await agent.update();
    actions = {}
    if (agent.step == 0) {
      
      actions = {
        faction: "AlphaStrike",
        spawns: [[20, 30]]
      }
    }
    console.log(JSON.stringify(actions))
    console.error(JSON.stringify(actions))
    }
});