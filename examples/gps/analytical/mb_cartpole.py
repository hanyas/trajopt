import autograd.numpy as np

import gym
from trajopt.gps import MBGPS

import warnings
warnings.filterwarnings("ignore")


# cartpole env
env = gym.make('Cartpole-TO-v1')
env._max_episode_steps = 10000
env.unwrapped._dt = 0.01

dm_state = env.observation_space.shape[0]
dm_act = env.action_space.shape[0]

horizon, nb_steps = 100, 500

env_sigma = env.unwrapped._sigma

state = np.zeros((dm_state, nb_steps + 1))
action = np.zeros((dm_act, nb_steps))

state[:, 0] = env.reset()
for t in range(nb_steps):
    solver = MBGPS(env, init_state=tuple([state[:, t], env_sigma]),
                   init_action_sigma=1., nb_steps=horizon, kl_bound=2.)
    trace = solver.run(nb_iter=10, verbose=False)

    _act = solver.ctl.sample(state[:, t], 0, stoch=False)
    action[:, t] = np.clip(_act, -env.ulim, env.ulim)
    state[:, t + 1], _, _, _ = env.step(action[:, t])

    print('Time Step:', t, 'Cost:', trace[-1])


import matplotlib.pyplot as plt

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
