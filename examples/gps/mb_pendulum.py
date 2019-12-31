import gym
from trajopt.gps import MBGPS

import warnings
warnings.filterwarnings("ignore")


# pendulum env
env = gym.make('Pendulum-TO-v1')
env._max_episode_steps = 500
env.unwrapped._dt = 0.01

alg = MBGPS(env, nb_steps=500,
            kl_bound=0.01,
            init_ctl_sigma=25.,
            activation={'shift': 250, 'mult': 0.025})

# run gps
trace = alg.run(nb_iter=200, verbose=True)

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

plt.subplot(3, 1, 1)
plt.plot(data['x'][0, :], '-b')
plt.subplot(3, 1, 2)
plt.plot(data['x'][1, :], '-b')

plt.subplot(3, 1, 3)
plt.plot(data['u'][0, :], '-g')

plt.show()
