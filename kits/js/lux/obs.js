const { Factory } = require("./factory");
const { Unit } = require("./unit");
const { Team } = require("./team");
const construct2DMat = (w, h, init=0) => {
  const mat = [];
  for (let i = 0; i < w; i++) {
    mat.push([]);
    for (let j = 0; j < h; j++) {
      mat[mat.length - 1].push(init);
    }
  }
  return mat;
};
const processObs = (agent, obs, step) => {
  let gameState = agent.gameState;
  if (step == 0) {
    gameState = obs["obs"];
    return gameState;
  } else {
    const envCfg = agent["env_cfg"];
    obs = obs["obs"];
    for (const k in obs) {
      if (k != "board" && k != "factories" && k !== "units") {
        gameState[k] = obs[k];
      }

      // process board
      for (const item of ["rubble", "lichen", "lichen_strains"]) {
        for (let k in obs["board"][item]) {
          const v = obs["board"][item][k];
          k = k.split(",");
          const x = parseInt(k[0]);
          const y = parseInt(k[1]);
          gameState["board"][item][y][x] = v;
        }
      }
    }
    factory_occupancy_map = construct2DMat(
      gameState["board"].rubble.length,
      gameState["board"].rubble[0].length,
      init = -1,
    );
    gameState["board"]["factory_occupancy_map"] = factory_occupancy_map;
    // process factories
    for (const player of [agent.player, agent.opp_player]) {
      gameState["factories"][player] = {};
      gameState["units"][player] = {};
      for (const k in obs["factories"][player]) {
        const f = obs["factories"][player][k];
        const factoryObj = new Factory(
          f.team_id,
          f.unit_id,
          f.power,
          f.pos,
          f.cargo,
          f.strain_id,
          envCfg
        );
        gameState["factories"][player][f.unit_id] = factoryObj;
        for (let dx = -1; dx <= 1; dx++) {
          for (let dy = -1; dy <= 1; dy++) {
            let x = f.pos[0] + dx;
            let y = f.pos[1] + dy;
            if (
              x < 0 ||
              (x >= factory_occupancy_map.length && y < 0) ||
              y[0] >= factory_occupancy_map[0].length
            ) {
              continue;
            }
            factory_occupancy_map[x][y] = f.strain_id;
          }
        }
      }
      for (const k in obs["units"][player]) {
        const u = obs["units"][player][k];
        const unitObj = new Unit(
          u.team_id,
          u.unit_id,
          u.power,
          u.pos,
          u.cargo,
          u.action_queue,
          u.unit_type,
          envCfg,
          envCfg.ROBOTS[u.unit_type]
        );
        gameState["units"][player][u.unit_id] = unitObj;
      }
    }

    let teams = {};
    for (let agent in obs.teams) {
      let team_data = obs.teams[agent];
      let faction = team_data["faction"];
      teams[agent] = new Team(
        team_data["team_id"],
        agent,
        faction,
        team_data["water"],
        team_data["metal"],
        team_data["factories_to_place"],
        team_data["factory_strains"],
        team_data["place_first"],
        team_data["bid"],
      );
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
};
module.exports = { processObs };
