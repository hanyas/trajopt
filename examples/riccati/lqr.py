#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: lqr.py
# @Date: 2019-06-16-18-38
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de


import gym
from trajopt.riccati import Riccati

# lqr task
env = gym.make('LQR-TO-v0')
env._max_episode_steps = 100

alg = Riccati(env, nb_steps=60,
              activation=range(60))

# run iLQR
trace = alg.run()

# plot forward pass
alg.plot()
