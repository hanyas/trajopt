import autograd.numpy as np

import gym
from trajopt.gps import MBGPS

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

np.random.seed(1337)

# lqr task
env = gym.make('LQR-TO-v0')
env._max_episode_steps = 100

env.seed(1337)

solver = MBGPS(env, nb_steps=60,
               init_state=env.init(),
               init_action_sigma=50.,
               kl_bound=1.)

trace = solver.run(nb_iter=50, verbose=True)

# plot dists
solver.plot()

# plot objective
plt.figure()
plt.plot(trace)
plt.show()
