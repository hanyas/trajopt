import autograd.numpy as np

import gym
from trajopt.gps import MFGPS

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

np.random.seed(1337)

# lqr task
env = gym.make('LQR-TO-v0')
env._max_episode_steps = 100000

env.seed(1337)

prior = {'K': 1e-6, 'psi': 1e-6, 'nu': 0.1}

alg = MFGPS(env, nb_steps=60,
            init_state=env.init(),
            init_action_sigma=100.,
            kl_bound=1.,
            dyn_prior=prior)

# run gps
trace = alg.run(nb_episodes=25, nb_iter=25, verbose=True)

# plot dists
alg.plot()

# execute and plot
data = alg.sample(25, stoch=True)

plt.figure()
for k in range(alg.dm_state):
    plt.subplot(alg.dm_state + alg.dm_act, 1, k + 1)
    plt.plot(data['x'][k, ...])

for k in range(alg.dm_act):
    plt.subplot(alg.dm_state + alg.dm_act, 1, alg.dm_state + k + 1)
    plt.plot(data['u'][k, ...])

plt.show()

# plot objective
plt.figure()
plt.plot(trace)
plt.show()
