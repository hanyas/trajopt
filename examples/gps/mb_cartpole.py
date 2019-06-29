#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: mb_cartpole.py
# @Date: 2019-06-16-18-38
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de


import gym
from trajopt.gps import MBGPS


# cartpole env
env = gym.make('Cartpole-TO-v0')
env._max_episode_steps = 200

alg = MBGPS(env, nb_steps=200,
            kl_bound=25.,
            init_ctl_sigma=1.0,
            activation='last')

# run gps
trace = alg.run()

# plot dists
alg.plot()

# plot objective
import matplotlib.pyplot as plt

plt.figure()
plt.plot(trace)
plt.show()
