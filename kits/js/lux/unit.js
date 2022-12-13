class Unit {
  constructor(teamId, unitId, power, pos, cargo, actionQueue, unitType, envCfg) {
    this.pos = pos;
    this.teamId = teamId;
    this.unitId = unitId;
    this.power = power;
    this.cargo = cargo;
    this.envCfg = envCfg;
    this.unitType = unitType;
    this.actionQueue = actionQueue;
  }

  // direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
  move(direction, repeat) {
    return [0, direction, 0, 0, repeat ? 1: 0];
  }

  transfer(transferDirection, transferResource, transferAmount, repeat) {
    if(!(transferDirection < 5 && transferDirection >= 0)) {
      console.error('invalid transferDirection', transferDirection);
      return;
    }
    if(!(transferResource < 5 && transferResource >= 0)) {
      console.error('invalid transferResource', transferResource);
      return;
    }
    return [1, transferDirection, transferResource, transferAmount, repeat ? 1: 0];
  }

  pickup(pickupResource, pickupAmount, repeat) {
    if(!(pickupResource < 5 && pickupResource >= 0)) {
      console.error('invalid pickupResource', pickupResource);
      return;
    }
    return [2, 0, pickupResource, pickupAmount, repeat ? 1: 0];
  }

  dig(repeat) {
    return [3, 0, 0, 0, repeat ? 1: 0];
  }

  recharge(x, repeat) {
    return [5, 0, 0, x, repeat ? 1: 0];
  }
}

module.exports = { Unit }
