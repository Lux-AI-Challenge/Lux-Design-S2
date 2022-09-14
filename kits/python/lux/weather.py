def get_weather_config(current_weather, env_cfg):
    if type(current_weather) == int:
        current_weather = env_cfg.WEATHER_ID_TO_NAME[current_weather]
    if current_weather == "MARS_QUAKE":
        return dict(power_gain_factor=1, power_loss_factor=1)
    elif current_weather == "COLD_SNAP":
        return dict(power_gain_factor=1, power_loss_factor=2)
    elif current_weather == "DUST_STORM":
        return dict(power_gain_factor=0.5, power_loss_factor=1)
    elif current_weather == "SOLAR_FLARE":
        return dict(power_gain_factor=2, power_loss_factor=1)
    return dict(power_gain_factor=1, power_loss_factor=1)