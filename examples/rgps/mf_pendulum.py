import autograd.numpy as np

import gym
from trajopt.rgps import MFRGPS

import warnings
warnings.filterwarnings("ignore")


np.random.seed(1337)

# pendulum task
env = gym.make('Pendulum-TO-v0')
env._max_episode_steps = 100
env.unwrapped.dt = 0.05

env.seed(1337)

prior = {'K': 1e-6}

alg = MFRGPS(env, nb_steps=100,
             policy_kl_bound=5.,
             param_kl_bound=1e0,
             init_state=env.init(),
             init_action_sigma=25.,
             action_penalty=1e-5,
             slew_rate=False,
             prior=prior)

trace = alg.run(nb_episodes=5, nb_iter=5, verbose=True)

import matplotlib.pyplot as plt

# plot objective
plt.figure()
plt.plot(trace)
plt.show()
