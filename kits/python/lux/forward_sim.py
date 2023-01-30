def forward_sim(full_obs, env_cfg, n=2):
    """
    Forward sims for `n` steps given the current full observation and env_cfg

    If forward sim leads to the end of a game, it won't return any additional observations, just the original one
    """
    from luxai_s2 import LuxAI_S2
    from luxai_s2.config import UnitConfig
    import copy
    agent = "player_0"
    env = LuxAI_S2(collect_stats=False, verbose=0)
    env.reset(seed=0)
    env.state = env.state.from_obs(full_obs, env_cfg)
    env.env_cfg = env.state.env_cfg
    env.env_cfg.verbose = 0
    env.env_steps = env.state.env_steps
    forward_obs = [full_obs]
    for _ in range(n):
        empty_actions = dict()
        for agent in env.agents:
            empty_actions[agent] = dict()
        if len(env.agents) == 0:
            # can't step any further
            return [full_obs]
        obs, _, _, _ = env.step(empty_actions)
        forward_obs.append(obs[agent])
    return forward_obs