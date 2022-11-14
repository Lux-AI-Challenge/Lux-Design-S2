const { getWeatherConfig } = require("./weather");

class Factory {
  constructor(teamId, unitId, power, pos, cargo, envCfg) {
    this.pos = pos
    this.teamId = teamId
    this.unitId = unitId
    this.power = power
    this.cargo = cargo
    this.envCfg = envCfg
  }

  buildHeavyPowerCost(gameState) {
    const currentWeather = gameState['weather_schedule'][gameState['real_env_steps']];
    const weatherConfig = getWeatherConfig(currentWeather, this.envCfg)
    const unitCfg = this.envCfg.ROBOTS["HEAVY"];
    return Math.ceil(unitCfg.POWER_COST * weatherConfig.powerlossFactor);
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
}

module.exports = { Factory }