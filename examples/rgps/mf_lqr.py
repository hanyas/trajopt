import numpy as np

import gym
from trajopt.rgps import MFRGPS
from trajopt.gps import MFGPS

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# lqr task
env = gym.make('LQR-TO-v1')
env.env._max_episode_steps = 100
env.unwrapped.perturb = True

prior = {'K': 1e-6}

# mass-damper
rgps = MFRGPS(env, nb_steps=100,
              policy_kl_bound=5.,
              param_kl_bound=1e1,
              init_state=env.init(),
              init_action_sigma=100.,
              prior=prior)

np.random.seed(1337)
env.seed(1337)

rgps.run(nb_iter=15, nb_learning_episodes=10,
         nb_evaluation_episodes=100, verbose=True)

gps = MFGPS(env, nb_steps=100,
            kl_bound=5.,
            init_state=env.init(),
            init_action_sigma=100.,
            dyn_prior=prior)

np.random.seed(1337)
env.seed(1337)

gps.run(nb_iter=15, nb_learning_episodes=10,
        nb_evaluation_episodes=100, verbose=True)

gps.plot()
