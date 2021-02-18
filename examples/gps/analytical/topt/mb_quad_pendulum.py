import autograd.numpy as np

import gym
from trajopt.gps import MBGPS

import warnings
warnings.filterwarnings("ignore")

np.random.seed(1337)

# pendulum env
env = gym.make('QuadPendulum-TO-v0')
env._max_episode_steps = 100
env.unwrapped.dt = 0.05

env.seed(1337)

solver = MBGPS(env, nb_steps=100,
               init_state=env.init(),
               init_action_sigma=5.0,
               kl_bound=10.,
               slew_rate=False,
               action_penalty=1e-5)

trace = solver.run(nb_iter=50, verbose=True)

# plot dists
solver.plot()

# plot objective
import matplotlib.pyplot as plt

plt.figure()
plt.plot(trace)
plt.show()
