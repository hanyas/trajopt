import autograd.numpy as np

import gym
from trajopt.ilqr import iLQR

import warnings
warnings.filterwarnings("ignore")


# lqr task
env = gym.make('LQR-TO-v0')
env._max_episode_steps = 100000

dm_state = env.observation_space.shape[0]
dm_act = env.action_space.shape[0]

horizon, nb_steps = 25, 100
state = np.zeros((dm_state, nb_steps + 1))
action = np.zeros((dm_act, nb_steps))

state[:, 0] = env.reset()
for t in range(nb_steps):
    solver = iLQR(env, init_state=state[:, t],
                  nb_steps=horizon)
    trace = solver.run(nb_iter=5, verbose=False)

    action[:, t] = solver.uref[:, 0]
    state[:, t + 1], _, _, _ = env.step(action[:, t])

    print('Time Step:', t, 'Cost:', trace[-1])


import matplotlib.pyplot as plt

plt.figure()

plt.subplot(3, 1, 1)
plt.plot(state[0, :], '-b')
plt.subplot(3, 1, 2)
plt.plot(state[1, :], '-b')

plt.subplot(3, 1, 3)
plt.plot(action[0, :], '-g')

plt.show()
