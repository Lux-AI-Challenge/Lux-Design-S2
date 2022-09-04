
const { exit } = require('process');
const kit = require('./lux/kit');
// const GAME_CONSTANTS = require('./lux/game_constants');
// const DIRECTIONS = GAME_CONSTANTS.DIRECTIONS;
// create a new agent
const agent = new kit.Agent();
// first initialize the agent, and then proceed to go in a loop waiting for updates and running the AI
agent.initialize().then(async () => {
  while (true) {
    /** Do not edit! **/
    // wait for updates
    await agent.update();
    actions = {}
    step = agent.step;
    if (step == 0) {
      actions = {
        faction: "AlphaStrike",
        spawns: [[20, 20 + Math.floor(Math.random() * 20)]]
      }
      // submit turn 0 actions
      console.log(JSON.stringify(actions));
      continue;
    }
    // various maps
    const rubble = agent.gameState["board"]["rubble"]

    // if ice[y][x] > 0, then there is an ice tile at (x, y)
    const ice = agent.gameState["board"]["ice"]
    // if ore[y][x] > 0, then there is an ore tile at (x, y)
    const ore = agent.gameState["board"]["ore"]

    // lichen[y][x] = amount of lichen at tile (x, y)
    const lichen = agent.gameState["board"]["lichen"]
    // lichenStrains[y][x] = the strain id of the lichen at tile (x, y). Each strain id is
    // associated with a single factory and cannot mix with other strains. 
    // factory.strain_id defines the factory's strain id
    const lichenStrains = agent.gameState["board"]["lichen_strains"]

    // units and factories for your team and the opposition team
    const units = agent.gameState["units"][agent.player]
    const oppUnits = agent.gameState["units"][agent.opp_player]
    const factories = agent.gameState["factories"][agent.player]
    const oppFactories = agent.gameState["factories"][agent.opp_player]

    for (const unit of Object.values(units)) {
      const unit_id = unit.unit_id
      direc = Math.floor(Math.random() * 5)
      // each unit can be given a plan of up to 10 actions to execute as fast as possible, limited by how much power the unit has
      // at each step you can give plans to up to 20 units
      actions[unit_id] = [[0, direc, 0, 0, 0]]
    }
    for (const factory of Object.values(factories)) {
      // each factory can be given a single action to perform each turn.
      const unit_id = factory["unit_id"]
      if (step % 4 == 0 && step > 1) {
        actions[unit_id] = Math.floor(Math.random() * 2);
      }
      else {
        actions[unit_id] = 2
      }
    }
    // submit actions
    console.log(JSON.stringify(actions));
    }
}).catch((err) => {
  console.error(err)
  process.exit(1)
});