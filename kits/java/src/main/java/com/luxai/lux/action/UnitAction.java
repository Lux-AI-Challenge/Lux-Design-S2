package com.luxai.lux.action;

import java.util.HashMap;
import java.util.Map;

public class UnitAction {

    public Map<String, Object> actions;

    public UnitAction() {
        this.actions = new HashMap<>();
    }

    public void add(String unit_id, Object action) {
        actions.put(unit_id, action);
    }

    public void addActions(UnitAction unitActions) {
        this.actions.putAll(unitActions.actions);
    }

}