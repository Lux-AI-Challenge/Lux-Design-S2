package com.luxai.lux.action;

import java.util.HashMap;
import java.util.Map;

public class UnitActions {

    public Map<String, Object> actions;

    public UnitActions() {
        this.actions = new HashMap<>();
    }

    public void add(String unit_id, Object action) {
        actions.put(unit_id, action);
    }

    public void addActions(UnitActions unitActions) {
        this.actions.putAll(unitActions.actions);
    }

}