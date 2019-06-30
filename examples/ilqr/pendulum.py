#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: pendulum.py
# @Date: 2019-06-16-18-38
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de


import gym
from trajopt.ilqr import iLQR

# pendulum env
env = gym.make('Pendulum-TO-v0')
env._max_episode_steps = 150

alg = iLQR(env, nb_steps=150,
           activation='last')

# run iLQR
trace = alg.run(nb_iter=25)

# plot forward pass
alg.plot()

# plot objective
import matplotlib.pyplot as plt

plt.figure()
plt.plot(trace)
plt.show()
