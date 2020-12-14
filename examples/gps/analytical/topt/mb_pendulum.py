import autograd.numpy as np

import gym
from trajopt.gps import MBGPS

import warnings
warnings.filterwarnings("ignore")

# pendulum env
env = gym.make('Pendulum-TO-v0')
env._max_episode_steps = 100
env.unwrapped.dt = 0.05

solver = MBGPS(env, nb_steps=100,
               init_state=env.init(),
               init_action_sigma=1.0,
               kl_bound=2., slew_rate=False,
               action_penalty=np.array([1e-5]))

trace = solver.run(nb_iter=25, verbose=True)

# plot dists
solver.plot()

# plot objective
import matplotlib.pyplot as plt

plt.figure()
plt.plot(trace)
plt.show()
