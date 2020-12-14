import autograd.numpy as np

import gym
from trajopt.ilqr import iLQR

import warnings
warnings.filterwarnings("ignore")


# pendulum env
env = gym.make('Cartpole-TO-v1')
env._max_episode_steps = 100000
env.unwrapped.dt = 0.02

dm_state = env.observation_space.shape[0]
dm_act = env.action_space.shape[0]

horizon, nb_steps = 50, 250

state = np.zeros((dm_state, nb_steps + 1))
action = np.zeros((dm_act, nb_steps))

state[:, 0] = env.reset()
for t in range(nb_steps):
    solver = iLQR(env, init_state=state[:, t],
                  nb_steps=horizon, action_penalty=np.array([1e-5]))
    trace = solver.run(nb_iter=10, verbose=False)

    action[:, t] = solver.uref[:, 0]
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
