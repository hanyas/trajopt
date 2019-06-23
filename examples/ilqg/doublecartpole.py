#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: doublecartpole.py
# @Date: 2019-06-16-18-38
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de


import gym
from trajopt.ilqg import iLQG

# double cartpole env
env = gym.make('DoubleCartpole-TO-v0')
env._max_episode_steps = 200

alg = iLQG(env, nb_steps=200, activation='last')

# run iLQG
alg.run()

# plot forward pass
alg.plot()
