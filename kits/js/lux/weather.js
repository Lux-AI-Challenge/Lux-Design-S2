function getWeatherConfig(currentWeather, envCfg) {
  const currentWeatherName = envCfg['WEATHER_ID_TO_NAME'][currentWeather];
  let powerGainFactor = 1;
  let powerlossFactor = 1;
  if(currentWeatherName === "COLD_SNAP") {
    powerlossFactor = envCfg["WEATHER"]["COLD_SNAP"]["POWER_CONSUMPTION"];
  } else if(currentWeatherName === "DUST_STORM") {
    powerGainFactor = envCfg["WEATHER"]["DUST_STORM"]["POWER_GAIN"];
  } else if(currentWeatherName === "SOLAR_FLARE") {
    powerGainFactor = envCfg["WEATHER"]["SOLAR_FLARE"]["POWER_GAIN"];
  }
  return {powerGainFactor, powerlossFactor};
}

module.exports = { getWeatherConfig }
