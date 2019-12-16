import autograd.numpy as np
import gym

import trajopt
from trajopt.envs.quanser.qube.ctrl import SwingUpCtrl


nb_episodes = 50
nb_steps = 350

obs, act = [], []

for i in range(nb_episodes):
    env = gym.make('QQube-v0')
    env._max_episode_steps = nb_steps

    ctl = SwingUpCtrl(ref_energy=0.04, energy_gain=25.0,
                      acc_max=5.0, alpha_max_pd_enable=20.,
                      pd_gain=[-0.5, 25.0, -0.25, 1.25])

    _obs = np.zeros((nb_steps + 1, 4))
    _act = np.zeros((nb_steps, 1))

    _obs[0, :] = env.reset()
    done = False
    for t in range(1, nb_steps + 1):
        u = ctl(_obs[t - 1, :])
        _obs[t, :], _, _, _dict = env.step(np.clip(u, -5., 5.))
        _act[t - 1, :] = _dict['u']

    obs.append(_obs)
    act.append(_act)

    env.close()


print(obs)

import matplotlib.pyplot as plt

plt.figure()

for _obs, _act in zip(obs, act):
    plt.subplot(5, 1, 1)
    plt.plot(_obs[:, 0], '-b')
    plt.subplot(5, 1, 2)
    plt.plot(_obs[:, 1], '-b')

    plt.subplot(5, 1, 3)
    plt.plot(_obs[:, 2], '-r')
    plt.subplot(5, 1, 4)
    plt.plot(_obs[:, 3], '-r')

    plt.subplot(5, 1, 5)
    plt.plot(_act[:, 0], '-g')

plt.show()


import pickle
data = {'obs': [_obs[:-1, :] for _obs in obs], 'act': act}
pickle.dump(data, open("energy_qube.pkl", "wb"))
