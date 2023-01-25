package com.luxai.utils;

public class MoveUtils {

    public static final int X = 0;
    public static final int Y = 1;

    public static final int MOVE_CENTER = 0;
    public static final int MOVE_DOWN = 1;
    public static final int MOVE_RIGHT = 2;
    public static final int MOVE_UP = 3;
    public static final int MOVE_LEFT = 4;

    public static final int MOVE_UNAVAILABLE = -1;

    public static int getManhattanDistance(int x1, int y1, int x2, int y2) {
        return (Math.abs(x1 - x2) + Math.abs(y1 - y2));
    }

    public static int getDirection(int xSource, int ySource, int xTarget, int yTarget) {
        int dx = xTarget - xSource;
        int dy = yTarget - ySource;

        if (dx == 0 && dy == 0)
            return MOVE_CENTER;

        if (Math.abs(dx) > Math.abs(dy)) {
            if (dx > 0)
                return MOVE_RIGHT;
            else
                return MOVE_LEFT;
        }
        else {
            if (dy > 0)
                return MOVE_UP;
            else
                return MOVE_DOWN;
        }
    }

    public static boolean isMyTurnToPlaceFactory(int step, boolean placeFirst) {
        if (placeFirst)
            return (step % 2 == 1);
        else
            return (step % 2 == 0);
    }

}
