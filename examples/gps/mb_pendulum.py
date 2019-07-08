#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: mb_pendulum.py
# @Date: 2019-06-16-18-38
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de


import gym
from trajopt.gps import MBGPS


# pendulum env
env = gym.make('Pendulum-TO-v0')
env._max_episode_steps = 150

alg = MBGPS(env, nb_steps=150,
            kl_bound=0.,
            init_ctl_sigma=4.,
            activation=range(100, 150))

# run gps
trace = alg.run(nb_iter=50)

# plot dists
alg.plot()

# plot objective
import matplotlib.pyplot as plt

plt.figure()
plt.plot(trace)
plt.show()
