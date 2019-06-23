#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: mb_pendulum.py
# @Date: 2019-06-16-18-38
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de


import gym
from trajopt.ilqg import iLQG

# pendulum env
env = gym.make('Pendulum-TO-v0')
env._max_episode_steps = 150

alg = iLQG(env, nb_steps=150, activation='last')

# run iLQG
alg.run()

# plot forward pass
alg.plot()
