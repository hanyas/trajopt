#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: lqr.py
# @Date: 2019-06-28-17-30
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de


import gym
from trajopt.elqr import eLQR

# lqr task
env = gym.make('LQR-TO-v0')
env._max_episode_steps = 50

alg = eLQR(env, nb_steps=50, activation='all')

# run eLQR
trace = alg.run()

# plot forward pass
alg.plot()

# plot objective
import matplotlib.pyplot as plt

plt.figure()
plt.plot(trace)
plt.show()

