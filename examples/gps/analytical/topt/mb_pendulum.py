import autograd.numpy as np

import gym
from trajopt.gps import MBGPS

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

np.random.seed(1337)

# pendulum env
env = gym.make('Pendulum-TO-v0')
env._max_episode_steps = 100
env.unwrapped.dt = 0.05

env.seed(1337)

solver = MBGPS(env, nb_steps=100,
               init_state=env.init(),
               init_action_sigma=5.0,
               kl_bound=1e1,
               slew_rate=False,
               action_penalty=1,
               activation={'mult': 1., 'shift': 80})

trace = solver.run(nb_iter=25, verbose=True)

# plot dists
solver.plot()

# plot objective
plt.figure()
plt.plot(trace)
plt.show()
