#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: mb_doublecartpole.py
# @Date: 2019-06-16-18-38
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de


import gym
from trajopt.gps import MBGPS


# double cartpole env
env = gym.make('DoubleCartpole-TO-v0')
env._max_episode_steps = 200

alg = MBGPS(env, nb_steps=200,
            kl_bound=10.,
            init_ctl_sigma=5.0,
            activation=range(150, 200))

# run gps
trace = alg.run()

# plot dists
alg.plot()

# plot objective
import matplotlib.pyplot as plt

plt.figure()
plt.plot(trace)
plt.show()
