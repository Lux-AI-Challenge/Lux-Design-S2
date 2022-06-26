from gym import spaces

from luxai2022.config import EnvConfig


def get_obs_space(config: EnvConfig, agent: int = 0):
    obs_space = dict()
    # observations regarding some meta information such as total units, resources etc.
    meta_obs_space = dict(
        weather=spaces.Box(
            low=0,
            high=len(config.WEATHER) + 1,
            shape=(config.max_episode_length, ),
            dtype=int,
        )
    )
    obs_space["meta"] = spaces.Dict(meta_obs_space)

    resources_obs_space = dict(
        ice=spaces.Box(low=0, high=1, shape=(config.map_size, config.map_size), dtype=int),
        ore=spaces.Box(low=0, high=1, shape=(config.map_size, config.map_size), dtype=int)
    )

    obs_space["resources"] = spaces.Dict(resources_obs_space)
    obs_space["rubble"] = spaces.Box(low=0, high=100, shape=(config.map_size, config.map_size), dtype=int)


    return spaces.Dict(obs_space)

if __name__ == "__main__":
    cfg = EnvConfig()
    obs_space = get_obs_space(cfg)
    import ipdb;ipdb.set_trace()