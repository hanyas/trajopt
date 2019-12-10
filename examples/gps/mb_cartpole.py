import gym
from trajopt.gps import MBGPS

import warnings
warnings.filterwarnings("ignore")


# cartpole env
env = gym.make('Cartpole-TO-v0')
env._max_episode_steps = 500

alg = MBGPS(env, nb_steps=500,
            kl_bound=0.05,
            init_ctl_sigma=10.0,
            activation=range(450, 500))

# run gps
trace = alg.run(nb_iter=100)

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

data = alg.sample(nb_episodes=1, stoch=False)

plt.figure()

plt.subplot(5, 1, 1)
plt.plot(data['x'][0, :], '-b')
plt.subplot(5, 1, 2)
plt.plot(data['x'][1, :], '-b')

plt.subplot(5, 1, 3)
plt.plot(data['x'][2, :], '-r')
plt.subplot(5, 1, 4)
plt.plot(data['x'][3, :], '-r')

plt.subplot(5, 1, 5)
plt.plot(data['u'][0, :], '-g')

plt.show()
