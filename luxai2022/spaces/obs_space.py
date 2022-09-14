from gym import spaces

from luxai2022.config import EnvConfig


def get_obs_space(config: EnvConfig, agent: int = 0):
    obs_space = dict()
    # observations regarding some meta information such as total units, resources etc.
    obs_space["weather"] = spaces.Box(
        low=0,
        high=len(config.WEATHER) + 1,
        shape=(config.max_episode_length,),
        dtype=int,
    )

    obs_space["day"] = spaces.Discrete(2)


    # rubble=self.rubble.copy(),
    # ore=self.ore.copy(),
    # ice=self.ice.copy(),
    # lichen=self.lichen.copy(),
    # lichen_strains=self.lichen_strains.copy(),
    # spawns=self.spawns.copy(),
    # factories_per_team=self.factories_per_team,
    board_obs_space = dict(
        ice=spaces.Box(low=0, high=1, shape=(config.map_size, config.map_size), dtype=int),
        ore=spaces.Box(low=0, high=1, shape=(config.map_size, config.map_size), dtype=int),
        rubble=spaces.Box(low=0, high=100, shape=(config.map_size, config.map_size), dtype=int),
        lichen=spaces.Box(low=0, high=99999, shape=(config.map_size, config.map_size), dtype=int),
        lichen_strains=spaces.Box(low=0, high=99999, shape=(config.map_size, config.map_size), dtype=int),
        factories_per_team=spaces.Discrete(99999)
        # spawns=spaces.Box(low=0, high=99999, shape=(config.map_size, config.map_size), dtype=int) # this array can change in size easily.
    )
    obs_space["board"] = spaces.Dict(board_obs_space)

    units_obs_space = dict()
    for i in range(2):
        # defines what each unit's info looks like since its variable
        # up to user to do any kind of padding or reshaping
        obs_dict = dict(
            power=spaces.Discrete(config.ROBOTS["HEAVY"].BATTERY_CAPACITY + 1),
            repeating_actions=spaces.Discrete(2),
            pos=spaces.Box(0, config.map_size, shape=(2,), dtype=int),
            cargo=spaces.Dict(
                ice=spaces.Discrete(config.ROBOTS["HEAVY"].CARGO_SPACE + 1),
                water=spaces.Discrete(config.ROBOTS["HEAVY"].CARGO_SPACE + 1),
                ore=spaces.Discrete(config.ROBOTS["HEAVY"].CARGO_SPACE + 1),
                metal=spaces.Discrete(config.ROBOTS["HEAVY"].CARGO_SPACE + 1),
            ),
        )
        if config.UNIT_ACTION_QUEUE_SIZE != 1:
            obs_dict["action_queue"] = spaces.MultiDiscrete([7, 5, 5, config.max_transfer_amount])
        units_obs_space[i] = spaces.Dict(obs_dict)

    obs_space["units"] = spaces.Dict(units_obs_space)

    factories_obs_space = dict()
    for i in range(2):
        # defines what each factory's info looks like since its variable
        # up to user to do any kind of padding or reshaping
        obs_dict = dict(
            power=spaces.Discrete(999999999),
            pos=spaces.Box(0, config.map_size, shape=(2,), dtype=int),
            cargo=spaces.Dict(
                ice=spaces.Discrete(config.ROBOTS["HEAVY"].CARGO_SPACE),
                water=spaces.Discrete(config.ROBOTS["HEAVY"].CARGO_SPACE),
                ore=spaces.Discrete(config.ROBOTS["HEAVY"].CARGO_SPACE),
                metal=spaces.Discrete(config.ROBOTS["HEAVY"].CARGO_SPACE),
            ),
        )
        factories_obs_space[i] = spaces.Dict(obs_dict)
    obs_space["factories"] = spaces.Dict(factories_obs_space)

    return spaces.Dict(obs_space)


if __name__ == "__main__":
    cfg = EnvConfig()
    obs_space = get_obs_space(cfg)
    import ipdb

    ipdb.set_trace()
