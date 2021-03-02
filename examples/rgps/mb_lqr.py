import gym
from trajopt.rgps import LRGPS
from trajopt.gps import MBGPS

import warnings
warnings.filterwarnings("ignore")

# lqr task
env = gym.make('LQR-TO-v1')
env.env._max_episode_steps = 100

# mass-damper
rgps = LRGPS(env, nb_steps=100,
             policy_kl_bound=0.25,
             param_kl_bound=25e2,
             init_state=env.init(),
             init_action_sigma=100.)

rgps.run(nb_iter=50, verbose=True)

gps = MBGPS(env, nb_steps=100,
            init_state=env.init(),
            init_action_sigma=100.,
            kl_bound=0.25)

gps.run(nb_iter=100, verbose=True)

rgps.compare(std_ctl=gps.ctl)
