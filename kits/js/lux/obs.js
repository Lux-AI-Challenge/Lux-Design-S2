const processObs = (agent, obs, step) => {
  let gameState = agent.gameState;
  if (step == 0) {
    gameState = obs["obs"];
    return gameState;
  } else {
    obs = obs["obs"];
    for (const k in obs) {
      if (k != 'board') {
        gameState[k] = obs[k]
      }
      for (const item of ["rubble", "lichen", "lichen_strains"]) {
        for (let k in obs["board"][item])  {
          const v = obs["board"][item][k]
          k = k.split(",")
          const x = parseInt(k[0])
          const y = parseInt(k[1])
          gameState["board"][item][y][x] = v;
        }
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