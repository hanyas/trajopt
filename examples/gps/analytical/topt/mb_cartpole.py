import autograd.numpy as np

import gym
from trajopt.gps import MBGPS

import warnings
warnings.filterwarnings("ignore")

# pendulum env
env = gym.make('Cartpole-TO-v0')
env._max_episode_steps = 250
env.unwrapped.dt = 0.02

solver = MBGPS(env, nb_steps=250,
               init_state=env.init(),
               init_action_sigma=1.0,
               kl_bound=1.,
               activation={'mult': 1.5, 'shift': 235})

trace = solver.run(nb_iter=50, verbose=True)

# plot dists
solver.plot()

# plot objective
import matplotlib.pyplot as plt

plt.figure()
plt.plot(trace)
plt.show()
