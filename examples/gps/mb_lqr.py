#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: mb_lqr.py
# @Date: 2019-06-16-18-38
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de


import gym
from trajopt.gps import MBGPS


# lqr task
env = gym.make('LQR-TO-v0')
env._max_episode_steps = 100

alg = MBGPS(env, nb_steps=100,
            kl_bound=10.,
            init_ctl_sigma=1.)

# run gps
for _ in range(1):
    alg.run()

# plot dists
alg.plot()

# execute and plot
nb_traj = 100
data = alg.sample(nb_traj, alg.nb_steps, stoch=False)

import matplotlib.pyplot as plt

plt.figure()

plt.subplot(3, 1, 1)
plt.plot(data['x'][0, ...])

plt.subplot(3, 1, 2)
plt.plot(data['x'][1, ...])

plt.subplot(3, 1, 3)
plt.plot(data['u'][0, ...])
plt.show()
