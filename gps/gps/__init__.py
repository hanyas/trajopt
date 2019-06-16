from .mbgps import MBGPS

from gym.envs.registration import register

register(
    id='LQR-TO-v0',
    entry_point='gps.envs:LQR',
    max_episode_steps=1000,
)

register(
    id='Pendulum-TO-v0',
    entry_point='gps.envs:Pendulum',
    max_episode_steps=1000,
)
