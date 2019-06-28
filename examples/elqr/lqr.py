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

alg = eLQR(env, nb_steps=50, activation='last')

alg.run()

alg.plot()
