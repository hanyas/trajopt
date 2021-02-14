import numpy as np

import gym
from trajopt.rgps import MFRGPS

import warnings
warnings.filterwarnings("ignore")

# np.random.seed(1337)

# lqr task
env = gym.make('LQR-TO-v0')
env._max_episode_steps = 100000

prior = {'K': 1e-6}

alg = MFRGPS(env, nb_steps=60,
             policy_kl_bound=10.,
             param_kl_bound=1000.,
             init_state=env.init(),
             init_action_sigma=100.,
             prior=prior)

trace = alg.run(nb_episodes=25, nb_iter=25,
                verbose=True, debug_dual=False)

alg.plot()

data = alg.sample(25, stoch=True)

import matplotlib.pyplot as plt

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
