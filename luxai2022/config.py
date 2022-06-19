class EnvConfig():
    def __init__(
        max_episode_length=1000,
        map_size=None,
        # TODO FILL IN ENV CONSTANTS
        env_constants=dict(
            FACTORY_PROCESSING_RATE_WATER=50,
            FACTORY_PROCESSING_RATE_METAL=50,
            ROBOTS=
                dict(
                    LIGHT=dict(
                        METAL_COST=10,
                        POWER_COST=50,
                        CARGO_SPACE=100
                    ),
                    HEAVY=dict(
                        METAL_COST=100,
                        POWER_COST=500,
                        CARGO_SPACE=1000
                    )
                )
        )
    ) -> None:
        pass