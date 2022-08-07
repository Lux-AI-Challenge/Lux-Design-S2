from gym import spaces

from luxai2022.config import EnvConfig


def get_obs_space(config: EnvConfig, agent: int = 0):
    obs_space = dict()
    # observations regarding some meta information such as total units, resources etc.
    meta_obs_space = dict(
        # TODO - maybe we make this a variable array
        weather=spaces.Box(
            low=0,
            high=len(config.WEATHER) + 1,
            shape=(config.max_episode_length,),
            dtype=int,
        )
    )
    obs_space["meta"] = spaces.Dict(meta_obs_space)

    obs_space["day"] = spaces.Discrete(2)

    resources_obs_space = dict(
        ice=spaces.Box(low=0, high=1, shape=(config.map_size, config.map_size), dtype=int),
        ore=spaces.Box(low=0, high=1, shape=(config.map_size, config.map_size), dtype=int),
    )

    obs_space["resources"] = spaces.Dict(resources_obs_space)
    obs_space["rubble"] = spaces.Box(low=0, high=100, shape=(config.map_size, config.map_size), dtype=int)

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
