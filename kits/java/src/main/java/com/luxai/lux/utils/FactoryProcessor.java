package com.luxai.lux.utils;

import com.luxai.lux.Environment;
import com.luxai.lux.Factory;
import com.luxai.lux.Obs;
import com.luxai.lux.action.UnitAction;

import java.util.Map;

public class FactoryProcessor {

    public static UnitAction getActions(Obs obs, Environment environment, String player) {
        UnitAction unitAction = new UnitAction();
        Map<String, Factory> myFactories = obs.factories.get(player);
        for (String unitId : myFactories.keySet()) {
            Factory factory = myFactories.get(unitId);
            if (factory.canBuildHeavy(environment))
                unitAction.add(factory.unit_id, Factory.BUILD_HEAVY);
        }
        return unitAction;
    }

}
