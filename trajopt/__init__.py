from gym.envs.registration import register

register(
    id='LQR-TO-v0',
    entry_point='trajopt.envs:LQRv0',
    max_episode_steps=1000,
)

register(
    id='LQR-TO-v1',
    entry_point='trajopt.envs:LQRv1',
    max_episode_steps=1000,
)

register(
    id='LQR-TO-v2',
    entry_point='trajopt.envs:LQRv2',
    max_episode_steps=1000,
)

register(
    id='Robot-TO-v0',
    entry_point='trajopt.envs:Robot',
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

register(
    id='DoublePendulum-TO-v0',
    entry_point='trajopt.envs:DoublePendulum',
    max_episode_steps=1000,
)

register(
    id='DoublePendulum-TO-v1',
    entry_point='trajopt.envs:DoublePendulumWithCartesianCost',
    max_episode_steps=1000,
)

register(
    id='QuadPendulum-TO-v0',
    entry_point='trajopt.envs:QuadPendulum',
    max_episode_steps=1000,
)

register(
    id='QuadPendulum-TO-v1',
    entry_point='trajopt.envs:QuadPendulumWithCartesianCost',
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
    id='QQube-v0',
    entry_point='trajopt.envs:Qube',
    max_episode_steps=10000,
    kwargs={'fs': 500.0, 'fs_ctrl': 100.0}
)

register(
    id='QQube-RR-v0',
    entry_point='trajopt.envs:QubeRR',
    max_episode_steps=10000,
    kwargs={'ip': '192.172.162.1', 'fs_ctrl': 100.0}
)

register(
    id='QCartpole-v0',
    entry_point='trajopt.envs:QCartpole',
    max_episode_steps=10000,
    kwargs={'fs': 500.0, 'fs_ctrl': 100.0, 'long_pole': False}
)

register(
     id='QCartpole-RR-v0',
     entry_point='trajopt.envs:QCartpoleRR',
     max_episode_steps=10000,
     kwargs={'ip': '192.172.162.1', 'fs_ctrl': 100.0}
)

register(
    id='QQube-TO-v0',
    entry_point='trajopt.envs:QubeTO',
    max_episode_steps=10000,
)

register(
    id='QQube-TO-v1',
    entry_point='trajopt.envs:QubeTOWithCartesianCost',
    max_episode_steps=10000,
)

register(
    id='QCartpole-TO-v0',
    entry_point='trajopt.envs:QCartpoleTO',
    max_episode_steps=10000,
    kwargs={'fs': 500.0, 'fs_ctrl': 100.0}
)
