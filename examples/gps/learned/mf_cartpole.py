import autograd.numpy as np

import gym
from trajopt.gps import MFGPS

import warnings
warnings.filterwarnings("ignore")


# pendulum task
env = gym.make('Cartpole-TO-v0')
env._max_episode_steps = 100
env.unwrapped.dt = 0.05

solver = MFGPS(env, nb_steps=100,
               init_state=env.init(),
               init_action_sigma=1.0,
               kl_bound=2.,
               activation={'mult': 1.5, 'shift': 95})

# run gps
trace = solver.run(nb_episodes=50, nb_iter=25, verbose=True)

# execute and plot
data = solver.sample(25, stoch=True)

import matplotlib.pyplot as plt

plt.figure()
for k in range(solver.dm_state):
    plt.subplot(solver.dm_state + solver.dm_act, 1, k + 1)
    plt.plot(data['x'][k, ...])

for k in range(solver.dm_act):
    plt.subplot(solver.dm_state + solver.dm_act, 1, solver.dm_state + k + 1)
    plt.plot(np.clip(data['u'][k, ...], - env.ulim, env.ulim))

plt.show()

# plot objective
plt.figure()
plt.plot(trace)
plt.show()
