import autograd.numpy as np

import gym
from trajopt.gps import MFGPS

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

np.random.seed(1337)

# cartpole task
env = gym.make('Cartpole-TO-v0')
env._max_episode_steps = 100
env.unwrapped.dt = 0.05

env.seed(1337)

prior = {'K': 1e-6, 'psi': 1e6, 'nu': 0.1}
# prior = {'K': 1e-6}

solver = MFGPS(env, nb_steps=100,
               init_state=env.init(),
               init_action_sigma=0.1,
               kl_bound=1e0,
               action_penalty=1e-3,
               prior=prior,
               activation={'mult': 1., 'shift': 80})

# run gps
trace = solver.run(nb_learning_episodes=25,
                   nb_iter=50, verbose=True)

# execute and plot
data = solver.rollout(25, stoch=True)

plt.figure()
for k in range(solver.dm_state):
    plt.subplot(solver.dm_state + solver.dm_act, 1, k + 1)
    plt.plot(data['x'][k, ...])

for k in range(solver.dm_act):
    plt.subplot(solver.dm_state + solver.dm_act, 1, solver.dm_state + k + 1)
    plt.plot(np.clip(data['u'], - env.ulim[:, None, None], env.ulim[:, None, None])[k, ...])

plt.show()

# plot objective
plt.figure()
plt.plot(trace)
plt.show()
