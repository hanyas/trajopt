#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: lqr.py
# @Date: 2019-06-16-18-38
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de


import gym
from trajopt.gps import MBGPS


# lqr task
env = gym.make('LQR-TO-v0')
env._max_episode_steps = 100

alg = MBGPS(env, nb_steps=100,
            kl_bound=50.,
            init_ctl_sigma=1.)

# run gps
for _ in range(2):
    alg.run()

alg.plot()
