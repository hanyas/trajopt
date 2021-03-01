import numpy as np

import gym
from trajopt.rgps import LRGPS
from trajopt.gps import MBGPS

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# lqr task
nominal_env = gym.make('LQR-TO-v1')
nominal_env.env._max_episode_steps = 100

dm_state = nominal_env.unwrapped.dm_state
dm_act = nominal_env.unwrapped.dm_act

# rgps = LRGPS(nominal_env, nb_steps=60,
#              policy_kl_bound=5.0,
#              param_kl_bound=50e3,
#              init_state=nominal_env.init(),
#              init_action_sigma=100.)

# mass-damper
rgps = LRGPS(nominal_env, nb_steps=100,
             policy_kl_bound=0.25,
             param_kl_bound=25e2,
             init_state=nominal_env.init(),
             init_action_sigma=100.)

rgps.run(nb_iter=25, verbose=True)

gps = MBGPS(nominal_env, nb_steps=100,
            init_state=nominal_env.init(),
            init_action_sigma=100.,
            kl_bound=5.)

gps.run(nb_iter=25, verbose=True)

rgps.compare(std_ctl=gps.ctl)
