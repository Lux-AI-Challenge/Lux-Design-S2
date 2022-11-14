const { Factory } = require("./factory");
const { Unit } = require("./unit");

const processObs = (agent, obs, step) => {
  let gameState = agent.gameState;
  if (step == 0) {
    gameState = obs["obs"];
    return gameState;
  } else {
    const envCfg = agent["env_cfg"];
    obs = obs["obs"];
    for (const k in obs) {
      if (k != 'board' && k != 'factories' && k !== 'units') {
        gameState[k] = obs[k]
      }

      // process board
      for (const item of ["rubble", "lichen", "lichen_strains"]) {
        for (let k in obs["board"][item])  {
          const v = obs["board"][item][k]
          k = k.split(",")
          const x = parseInt(k[0])
          const y = parseInt(k[1])
          gameState["board"][item][y][x] = v;
        }
      }

      // process factories
      gameState['factories'][agent.player] = {};
      for (const k in obs["factories"][agent.player]) {
          const f = obs["factories"][agent.player][k];
          const factoryObj = new Factory(f.team_id, f.unit_id, f.power, f.pos, f.cargo, envCfg) ;
          gameState['factories'][agent.player][f.unit_id] = factoryObj;
      }
      gameState['factories'][agent.opp_player] = {};
      for (const k in obs["factories"][agent.opp_player]) {
        const f = obs["factories"][agent.opp_player][k];
        const factoryObj = new Factory(f.team_id, f.unit_id, f.power, f.pos, f.cargo, envCfg) ;
        gameState['factories'][agent.opp_player][f.unit_id] = factoryObj;
      }

      // process units
      gameState['units'][agent.player] = {};
      for (const k in obs["units"][agent.player]) {
          const u = obs["units"][agent.player][k];
          const unitObj = new Unit(u.team_id, u.unit_id, u.power, u.pos, u.cargo, u.action_queue, envCfg) ;
          gameState['units'][agent.player][u.unit_id] = unitObj;
      }
      gameState['units'][agent.opp_player] = {};
      for (const k in obs["units"][agent.opp_player]) {
        const u = obs["units"][agent.opp_player][k];
        const unitObj = new Unit(u.team_id, u.unit_id, u.power, u.pos, u.cargo, u.action_queue, envCfg) ;
        gameState['units'][agent.opp_player][u.unit_id] = unitObj;
      }
    }
    return gameState;
    // # use delta changes to board to update game state
    //     obs = from_json(obs)
    //     for k in obs:
    //         if k != 'board':
    //             game_state[agent][k] = obs[k]
    //     for item in ["rubble", "lichen", "lichen_strains"]:
    //         for k, v in obs["board"][item].items():
    //             k = k.split(",")
    //             x, y = int(k[0]), int(k[1])
    //             game_state[agent]["board"][item][y, x] = v
  }
}
module.exports = { processObs }