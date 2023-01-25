package com.luxai.lux.action;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.luxai.objectmapper.Mapper;

import java.util.HashMap;
import java.util.Map;

public class UnitActions {

    public static final int NO_REPEAT_ACTION = 0;
    public static final int REPEAT_ACTION = 1;

    private final Map<String, Object> actions;

    public UnitActions() {
        this.actions = new HashMap<>();
    }

    public void add(String unitId, Object action) {
        actions.put(unitId, action);
    }

    public void addActions(UnitActions unitActions) {
        this.actions.putAll(unitActions.actions);
    }

    public String toSystemResponse() throws JsonProcessingException {
        if (actions.size() > 0) return Mapper.getJson(actions);

        return null;
    }
}