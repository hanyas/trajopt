import autograd.numpy as np

import gym
from trajopt.ilqr import iLQR

import warnings
warnings.filterwarnings("ignore")


# double cartpole env
env = gym.make('DoubleCartpole-TO-v1')
env._max_episode_steps = 100000
env.unwrapped._dt = 0.05

dm_state = env.observation_space.shape[0]
dm_act = env.action_space.shape[0]

horizon, nb_steps = 25, 100

state = np.zeros((dm_state, nb_steps + 1))
action = np.zeros((dm_act, nb_steps))
init_action = np.zeros((dm_act, horizon))

state[:, 0] = env.reset()
for t in range(nb_steps):
    solver = iLQR(env, init_state=state[:, t],
                  init_action=None, nb_steps=horizon)
    trace = solver.run(nb_iter=10, verbose=False)

    _nominal_action = solver.uref
    action[:, t] = _nominal_action[:, 0]
    state[:, t + 1], _, _, _ = env.step(action[:, t])

    init_action = np.hstack((_nominal_action[:, 1:], np.zeros((dm_act, 1))))
    print('Time Step:', t, 'Cost:', trace[-1])


import matplotlib.pyplot as plt

plt.figure()

plt.subplot(7, 1, 1)
plt.plot(state[0, :], '-b')
plt.subplot(7, 1, 2)
plt.plot(state[1, :], '-b')
plt.subplot(7, 1, 3)
plt.plot(state[2, :], '-b')

plt.subplot(7, 1, 4)
plt.plot(state[3, :], '-r')
plt.subplot(7, 1, 5)
plt.plot(state[4, :], '-r')
plt.subplot(7, 1, 6)
plt.plot(state[5, :], '-r')

plt.subplot(7, 1, 7)
plt.plot(action[0, :], '-g')

plt.show()
