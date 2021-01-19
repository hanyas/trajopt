import numpy as np

import gym
from trajopt.rgps import MFRGPS

import warnings
warnings.filterwarnings("ignore")

# np.random.seed(1337)

# pendulum task
env = gym.make('Pendulum-TO-v0')
env._max_episode_steps = 100
env.unwrapped.dt = 0.05

prior = {'K': 1e-6}

alg = MFRGPS(env, nb_steps=100,
             policy_kl_bound=1.,
             param_kl_bound=10.,
             init_state=env.init(),
             init_action_sigma=1.,
             action_penalty=np.array([1e-5]),
             slew_rate=False,
             prior=prior)

trace = alg.run(nb_episodes=50, nb_iter=25, verbose=True)

alg.plot()

data = alg.sample(25, stoch=True)

import matplotlib.pyplot as plt

plt.figure()
for k in range(alg.dm_state):
    plt.subplot(alg.dm_state + alg.dm_act, 1, k + 1)
    plt.plot(data['x'][k, ...])

for k in range(alg.dm_act):
    plt.subplot(alg.dm_state + alg.dm_act, 1, alg.dm_state + k + 1)
    plt.plot(np.clip(data['u'][k, ...], - env.ulim, env.ulim))

plt.show()

# plot objective
plt.figure()
plt.plot(trace)
plt.show()
