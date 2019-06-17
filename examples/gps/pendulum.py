#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: lqr.py
# @Date: 2019-06-16-18-38
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de


import gym
from trajopt.gps.mbgps import MBGPS

# pendulum env
env = gym.make('Pendulum-TO-v0')
env._max_episode_steps = 150

alg = MBGPS(env, nb_steps=150,
            kl_bound=200.,
            init_ctl_sigma=25.)

# run gps
for _ in range(25):
    alg.run()

alg.plot()
