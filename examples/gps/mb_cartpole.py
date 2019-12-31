import gym
from trajopt.gps import MBGPS

import warnings
warnings.filterwarnings("ignore")

# cartpole env
env = gym.make('Cartpole-TO-v0')
env._max_episode_steps = 500
env.unwrapped._dt = 0.01

alg = MBGPS(env, nb_steps=500,
            kl_bound=0.01,
            init_ctl_sigma=50.,
            activation={'shift': 450, 'mult': 0.5})

# run gps
trace = alg.run(nb_iter=150, verbose=True)

# plot dists
alg.plot()

# plot objective
import matplotlib.pyplot as plt

plt.figure()
plt.plot(trace)
plt.show()

# sample and plot one trajectory
data = alg.sample(nb_episodes=1, stoch=False)

plt.figure()

plt.subplot(6, 1, 1)
plt.plot(data['x'][0, :], '-b')
plt.subplot(6, 1, 2)
plt.plot(data['x'][1, :], '-b')

plt.subplot(6, 1, 3)
plt.plot(data['x'][2, :], '-r')
plt.subplot(6, 1, 4)
plt.plot(data['x'][3, :], '-r')

plt.subplot(6, 1, 5)
plt.plot(data['u'][0, :], '-g')

plt.subplot(6, 1, 6)
plt.plot(data['c'], '-k')

plt.show()
