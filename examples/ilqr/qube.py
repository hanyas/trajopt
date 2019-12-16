import gym
from trajopt.ilqr import iLQR

import warnings
warnings.filterwarnings("ignore")


# pendulum env
env = gym.make('QQube-TO-v1')
env._max_episode_steps = 500

alg = iLQR(env, nb_steps=500,
           activation={'shift': 250, 'mult': 0.01})

# run gps
trace = alg.run(nb_iter=5, verbose=True)

# plot dists
alg.plot()

# plot objective
import matplotlib.pyplot as plt

plt.figure()
plt.plot(trace)
plt.show()

state, action, _ = alg.forward_pass(ctl=alg.ctl, alpha=1.)

plt.figure()

plt.subplot(5, 1, 1)
plt.plot(state[0, :], '-b')
plt.subplot(5, 1, 2)
plt.plot(state[1, :], '-b')

plt.subplot(5, 1, 3)
plt.plot(state[2, :], '-r')
plt.subplot(5, 1, 4)
plt.plot(state[3, :], '-r')

plt.subplot(5, 1, 5)
plt.plot(action[0, :], '-g')

plt.show()
