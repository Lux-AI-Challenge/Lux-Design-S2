class Factory {
  constructor(teamId, unitId, power, pos, cargo, strainId, envCfg) {
    this.pos = pos;
    this.teamId = teamId;
    this.unitId = unitId;
    this.power = power;
    this.cargo = cargo;
    this.strainId = strainId;
    this.envCfg = envCfg;
  }

  buildHeavyPowerCost(gameState) {
    const unitCfg = this.envCfg.ROBOTS["HEAVY"];
    return unitCfg.POWER_COST;
  }
  buildHeavyMetalCost(gameState) {
    const unitCfg = this.envCfg.ROBOTS["HEAVY"];
    return unitCfg.METAL_COST;
  }

  canBuildHeavy(gameState) {
    return (
      this.power >= this.buildHeavyPowerCost(gameState) &&
      this.cargo.metal >= this.buildHeavyMetalCost(gameState)
    );
  }

  buildHeavy() {
    return 1;
  }

  buildLight() {
    return 0;
  }
  waterCost(gameState) {
    // note that this is simply an estimate, if lichen is cutoff this will be overestimated.
    const strains = gameState["board"]["lichen_strains"];
    let ownedLichenTiles = 0;
    for (let x = 0; x < strains.length; x++) {
      for (let y = 0; y < strains[x].length; y++) {
        if (strains[x][y] == this.strainId) {
          ownedLichenTiles += 1;
        }
      }
    }
    return Math.ceil(ownedLichenTiles / this.envCfg.LICHEN_WATERING_COST_FACTOR);
  }
  canWater(gameState) {
    return this.cargo.water >= this.waterCost(gameState);
  }
  water() {
    return 2;
  }
}

module.exports = { Factory }
