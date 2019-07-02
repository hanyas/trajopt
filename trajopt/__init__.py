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
    entry_point='trajopt.envs:PendulumWithCartesianCost',
    max_episode_steps=1000,
)

# pendulum in operational space
register(
    id='Pendulum-TO-v2',
    entry_point='trajopt.envs:PendulumWithCartesianObservation',
    max_episode_steps=1000,
)

register(
    id='Cartpole-TO-v0',
    entry_point='trajopt.envs:Cartpole',
    max_episode_steps=1000,
)

register(
    id='Cartpole-TO-v1',
    entry_point='trajopt.envs:CartpoleWithCartesianCost',
    max_episode_steps=1000,
)

register(
    id='DoubleCartpole-TO-v0',
    entry_point='trajopt.envs:DoubleCartpole',
    max_episode_steps=1000,
)

register(
    id='DoubleCartpole-TO-v1',
    entry_point='trajopt.envs:DoubleCartpoleWithCartesianCost',
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

register(
    id='Quanser-Qube-TO-v0',
    entry_point='trajopt.envs:Qube',
    max_episode_steps=300,
    kwargs={'fs': 500.0, 'fs_ctrl': 100.0}
)

register(
    id='Quanser-QubeRR-TO-v0',
    entry_point='trajopt.envs:QubeRR',
    max_episode_steps=300,
    kwargs={'ip': '192.172.162.1', 'fs_ctrl': 100.0}
)

register(
    id='Quanser-Cartpole-TO-v0',
    entry_point='trajopt.envs:QCartpole',
    max_episode_steps=10000,
    kwargs={'fs': 500.0, 'fs_ctrl': 500.0, 'long_pole': False}
)

register(
    id='Quanser-CartpoleRR-TO-v0',
    entry_point='trajopt.envs:QCartpoleRR',
    max_episode_steps=10000,
    kwargs={'ip': '192.172.162.1', 'fs_ctrl': 500.0}
)
