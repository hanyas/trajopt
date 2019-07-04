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
env._max_episode_steps = 60

alg = MBGPS(env, nb_steps=60,
            kl_bound=100.,
            init_ctl_sigma=100.,
            activation=range(60))

# run gps
trace = alg.run()

# plot dists
alg.plot()

# plot objective
import matplotlib.pyplot as plt

plt.figure()
plt.plot(trace)
plt.show()
