package com.luxai.lux;

import com.luxai.lux.weather.ColdSnap;
import com.luxai.lux.weather.DustStorm;
import com.luxai.lux.weather.MarsQuake;
import com.luxai.lux.weather.SolarFlare;

public class Weather {

    public MarsQuake MARS_QUAKE;
    public ColdSnap COLD_SNAP;
    public DustStorm DUST_STORM;
    public SolarFlare SOLAR_FLARE;

    public static double getGainFactor(int weatherCode, Environment environment) {
        switch (getWeatherName(weatherCode, environment)) {
            case "DUST_STORM":
                return environment.WEATHER.DUST_STORM.POWER_GAIN;
            case "SOLAR_FLARE":
                return environment.WEATHER.SOLAR_FLARE.POWER_GAIN;
            case "NONE":
            case "MARS_QUAKE":
            case "COLD_SNAP":
            default:
                return 1;
        }
    }

    public static double powerLossFactor(int weatherCode, Environment environment) {
        switch (getWeatherName(weatherCode, environment)) {
            case "COLD_SNAP":
                return environment.WEATHER.COLD_SNAP.POWER_CONSUMPTION;
            case "DUST_STORM":
            case "SOLAR_FLARE":
            case "NONE":
            case "MARS_QUAKE":
            default:
                return 1;
        }
    }

    public static String getWeatherName(int weatherCode, Environment environment) {
        return environment.WEATHER_ID_TO_NAME.get(weatherCode);
    }
}
