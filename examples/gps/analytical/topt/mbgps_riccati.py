import numpy as np

import gym

from trajopt.gps import MBGPS
from trajopt.riccati import Riccati

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# lqr task
env = gym.make('LQR-TO-v0')
env.env._max_episode_steps = 100

dm_state = env.unwrapped.dm_state
dm_act = env.unwrapped.dm_act

mbgps = MBGPS(env, nb_steps=100,
              init_state=env.init(),
              init_action_sigma=100.,
              kl_bound=5.)

mbgps.run(nb_iter=15, verbose=True)

riccati = Riccati(env, nb_steps=100,
                  init_state=env.init())

riccati.run()

np.random.seed(1337)
env.seed(1337)
gps_data = mbgps.rollout(250, stoch=False)

np.random.seed(1337)
env.seed(1337)
riccati_data = riccati.rollout(250)

print('GPS Cost: ', np.mean(np.sum(gps_data['c'], axis=0)),
      ', Riccati Cost', np.mean(np.sum(riccati_data['c'], axis=0)))

plt.figure(figsize=(6, 12))
plt.suptitle("LQR Mean Traj.: Riccati vs GPS")

for i in range(dm_state):
    plt.subplot(dm_state + dm_act, 1, i + 1)
    plt.plot(riccati.xref[i, ...], color='k')
    plt.plot(mbgps.xdist.mu[i, ...], color='r', linestyle='None', marker='o', markersize=3)

for i in range(dm_act):
    plt.subplot(dm_state + dm_act, 1, dm_state + i + 1)
    plt.plot(riccati.uref[i, ...], color='k')
    plt.plot(mbgps.udist.mu[i, ...], color='r', linestyle='None', marker='o', markersize=3)

plt.show()

plt.figure(figsize=(6, 12))
plt.suptitle("LQR Controller: Riccati vs GPS")

for i in range(dm_state):
    plt.subplot(dm_state + dm_act, 1, i + 1)
    plt.plot(mbgps.ctl.K[0, i, ...], color='r', linestyle='None', marker='o', markersize=3)
    plt.plot(riccati.ctl.K[0, i, ...], color='k')

for i in range(dm_act):
    plt.subplot(dm_state + dm_act, 1, dm_state + i + 1)
    plt.plot(mbgps.ctl.kff[i, ...], color='r', linestyle='None', marker='o', markersize=3)
    plt.plot(riccati.ctl.kff[i, ...], color='k')

plt.show()

plt.figure(figsize=(12, 12))
plt.suptitle("LQR Sample Traj.: Riccati vs GPS")

for i, j in zip(range(dm_state), range(1, 2 * dm_state, 2)):
    plt.subplot(dm_state + dm_act, 2, j)
    plt.plot(riccati_data['x'][i, ...])

    plt.subplot(dm_state + dm_act, 2, j + 1)
    plt.plot(gps_data['x'][i, ...])

for i, j in zip(range(dm_act), range(2 * dm_state + 1, 2 * dm_state + 2 * dm_act, 2)):
    plt.subplot(dm_state + dm_act, 2, j)
    plt.plot(riccati_data['u'][i, ...])

    plt.subplot(dm_state + dm_act, 2, j + 1)
    plt.plot(gps_data['u'][i, ...])

plt.show()
