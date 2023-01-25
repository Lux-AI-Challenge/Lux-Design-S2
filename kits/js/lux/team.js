class Team {
  constructor(team_id, agent, faction = null, water = 0, metal = 0, factories_to_place = 0, factory_strains = [], place_first = false, bid = 0) {
    this.faction = faction;
    this.team_id = team_id;
    this.agent = agent;
    this.water = water;
    this.metal = metal;
    this.factories_to_place = factories_to_place;
    this.factory_strains = factory_strains;
    this.place_first = place_first;
  }
  state_dict() {
    return {
      team_id: this.team_id,
      faction: this.faction.name,
      water: this.init_water,
      metal: this.init_metal,
      factories_to_place: this.factories_to_place,
      factory_strains: this.factory_strains,
      place_first: this.place_first
    };
  }
}

module.exports = {Team}