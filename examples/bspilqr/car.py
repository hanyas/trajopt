#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: car
# @Date: 2019-06-27-13-41
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de


import gym
from trajopt.bspilqr import BSPiLQR

# car task
env = gym.make('Car-TO-v0')
env._max_episode_steps = 25

alg = BSPiLQR(env, nb_steps=25, activation='last')

# run belief-space ilqr
trace = alg.run(nb_iter=5)

# plot forward pass
alg.plot()

# plot objective
import matplotlib.pyplot as plt

plt.figure()
plt.plot(trace)
plt.show()
