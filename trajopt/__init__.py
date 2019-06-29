from gym.envs.registration import register

register(
    id='LQR-TO-v0',
    entry_point='trajopt.envs:LQR',
    max_episode_steps=1000,
)

register(
    id='Pendulum-TO-v0',
    entry_point='trajopt.envs:Pendulum',
    max_episode_steps=1000,
)

register(
    id='Pendulum-TO-v1',
    entry_point='trajopt.envs:TrigPendulum',
    max_episode_steps=1000,
)

register(
    id='Cartpole-TO-v0',
    entry_point='trajopt.envs:Cartpole',
    max_episode_steps=1000,
)

register(
    id='DoubleCartpole-TO-v0',
    entry_point='trajopt.envs:DoubleCartpole',
    max_episode_steps=1000,
)

register(
    id='LightDark-TO-v0',
    entry_point='trajopt.envs:LightDark',
    max_episode_steps=1000,
)

register(
    id='Car-TO-v0',
    entry_point='trajopt.envs:Car',
    max_episode_steps=1000,
)
