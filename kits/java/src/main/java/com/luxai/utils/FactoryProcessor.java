package com.luxai.utils;

import com.luxai.lux.Environment;
import com.luxai.lux.Factory;
import com.luxai.lux.Obs;
import com.luxai.lux.action.UnitActions;

import java.util.Map;

public class FactoryProcessor {

    public static UnitActions getActions(Obs obs, Environment environment, String player) {
        UnitActions unitActions = new UnitActions();
        Map<String, Factory> myFactories = obs.playerToFactories.get(player);
        for (Factory factory : myFactories.values()) {
            if (factory.canBuildLight(environment))
                unitActions.add(factory.unitId, Factory.BUILD_LIGHT);
        }
        return unitActions;
    }

}
