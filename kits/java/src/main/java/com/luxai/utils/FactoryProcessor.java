package com.luxai.utils;

import com.luxai.lux.Environment;
import com.luxai.lux.Factory;
import com.luxai.lux.Obs;
import com.luxai.lux.action.UnitActions;

import java.util.Map;

public class FactoryProcessor {

    public static UnitActions getActions(Obs obs, Environment environment, String player) {
        UnitActions unitActions = new UnitActions();
        Map<String, Factory> myFactories = obs.factories.get(player);
        for (String unitId : myFactories.keySet()) {
            Factory factory = myFactories.get(unitId);
            if (factory.canBuildHeavy(environment))
                unitActions.add(factory.unit_id, Factory.BUILD_HEAVY);
        }
        return unitActions;
    }

}
