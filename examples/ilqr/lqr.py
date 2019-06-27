#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: lqr.py
# @Date: 2019-06-16-18-38
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de


import gym
from trajopt.ilqr import iLQR

# lqr task
env = gym.make('LQR-TO-v0')
env._max_episode_steps = 50

alg = iLQR(env, nb_steps=50,
           activation='all')

# run iLQR
alg.run()

# plot forward pass
alg.plot()
