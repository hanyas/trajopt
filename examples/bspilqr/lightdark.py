#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: lightdark.py
# @Date: 2019-06-27-13-40
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

import gym
from trajopt.bspilqr import BSPiLQR

# light dark task
env = gym.make('LightDark-TO-v0')
env._max_episode_steps = 25

alg = BSPiLQR(env, nb_steps=25, activation='all')

alg.run()

alg.plot()
