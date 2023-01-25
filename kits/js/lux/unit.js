const moveDeltas = [
  [0, 0],
  [0, -1],
  [1, 0],
  [0, 1],
  [-1, 0],
];
class Unit {
  constructor(
    teamId,
    unitId,
    power,
    pos,
    cargo,
    actionQueue,
    unitType,
    envCfg,
    unitCfg
  ) {
    this.teamId = teamId;
    this.unitId = unitId;
    this.unitType = unitType;
    this.pos = pos;
    this.power = power;
    this.cargo = cargo;
    this.envCfg = envCfg;
    this.unitCfg = unitCfg;
    this.actionQueue = actionQueue;
    if (this.teamId == 0) {
      this.agentId = "player_0";
    } else {
      this.agentId = "player_1";
    }
  }

  actionQueueCost(gameState) {
    return this.unitCfg.ACTION_QUEUE_POWER_COST
  }

  // direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
  move(direction, repeat, n = 1) {
    return [0, direction, 0, 0, repeat ? 1 : 0, n];
  }
  moveCost(gameState, direction) {
    const board = gameState.board;
    const targetPos = [
      this.pos[0] + moveDeltas[direction][0],
      this.pos[1] + moveDeltas[direction][1],
    ];
    if (
      targetPos[0] < 0 ||
      targetPos[1] < 0 ||
      targetPos[1] >= board.rubble.length ||
      targetPos[0] >= board.rubble[0].length
    ) {
      console.error("Warning, tried to get move cost for going off the map");
      return null;
    }
    const factoryThere =
      board.factory_occupancy_map[targetPos[0]][targetPos[1]];
    
    const factory_strains = gameState.teams[this.agentId].factory_strains;
    if (factory_strains.indexOf(factoryThere) == -1 && factoryThere != -1) {
      console.error(
        "Warning, tried to get move cost for going onto a opposition factory"
      );
      return null;
    }
    const rubble_at_target = board.rubble[targetPos[0]][targetPos[1]];

    return Math.floor(
      this.unitCfg.MOVE_COST +
        this.unitCfg.RUBBLE_MOVEMENT_COST * rubble_at_target
    );
  }

  transfer(transferDirection, transferResource, transferAmount, repeat, n=1) {
    if (!(transferDirection < 5 && transferDirection >= 0)) {
      console.error("invalid transferDirection", transferDirection);
      return;
    }
    if (!(transferResource < 5 && transferResource >= 0)) {
      console.error("invalid transferResource", transferResource);
      return;
    }
    return [
      1,
      transferDirection,
      transferResource,
      transferAmount,
      repeat ? 1 : 0,
      n
    ];
  }

  pickup(pickupResource, pickupAmount, repeat, n=1) {
    if (!(pickupResource < 5 && pickupResource >= 0)) {
      console.error("invalid pickupResource", pickupResource);
      return;
    }
    return [2, 0, pickupResource, pickupAmount, repeat ? 1 : 0, n];
  }
  digCost() {
    return this.unitCfg.DIG_COST
  }
  dig(repeat, n=1) {
    return [3, 0, 0, 0, repeat ? 1 : 0, n];
  }

  selfDestructCost(gameState) {
    return this.unitCfg.SEF_DESTRUCT_COST
  }
  selfDestruct(repeat, n= 1) {
    return [4, 0, 0, 0, repeat ? 1 : 0, n]
  }

  recharge(x, repeat, n=1) {
    return [5, 0, 0, x, repeat ? 1 : 0, n];
  }
}

module.exports = { Unit };
