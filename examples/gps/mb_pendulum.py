import gym
from trajopt.gps import MBGPS

import warnings
warnings.filterwarnings("ignore")


# pendulum env
env = gym.make('Pendulum-TO-v0')
env._max_episode_steps = 500

alg = MBGPS(env, nb_steps=500,
            kl_bound=0.1,
            init_ctl_sigma=10.0,
            activation=range(450, 500))

# run gps
trace = alg.run(nb_iter=200)

# plot dists
alg.plot()

# plot objective
import matplotlib.pyplot as plt

plt.figure()
plt.plot(trace)
plt.show()

# sample and plot one trajectory
import numpy as np

alg.ctl.sigma = np.ones_like(alg.ctl.sigma) * 1e-2

data = alg.sample(nb_episodes=1, stoch=True)

plt.figure()

plt.subplot(3, 1, 1)
plt.plot(data['x'][0, :], '-b')
plt.subplot(3, 1, 2)
plt.plot(data['x'][1, :], '-b')

plt.subplot(3, 1, 3)
plt.plot(data['u'][0, :], '-g')

plt.show()
