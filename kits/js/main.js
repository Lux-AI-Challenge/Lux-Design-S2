
const { exit } = require('process');
const kit = require('./agent');
// const GAME_CONSTANTS = require('./lux/game_constants');
// const DIRECTIONS = GAME_CONSTANTS.DIRECTIONS;
// create a new agent
const agent = new kit.Agent();

agent.initialize().then(async () => {
  while (true) {
    /** Do not edit! **/
    // wait for updates
    await agent.update();
    actions = {}
    step = agent.step;
    if (step <= agent.gameState["board"]["factories_per_team"] + 1) {
      actions = agent.earlySetup(step);
      // submit turn 0 actions
      console.log(JSON.stringify(actions));
    } else {
      actions = agent.act(agent.real_env_step);
      // submit turn 0 actions
      console.log(JSON.stringify(actions));
    }
  }
}).catch((err) => {
  console.error(err)
  process.exit(1)
});